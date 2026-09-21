"""SVS レンダリング: 楽譜 → 波形。

パイプライン: score_io（楽譜→フレーズ）→ duration_postprocess（音素列＋フレーム長）
→ PrettyPitch F0 モデル（infer.synthesize_f0, 2段ピッチ補正込み）
→ LeapSinger 音響モデル（mel）→ NHVSing v3.2.1 ボコーダ ONNX（mel→波形）
→ フレーズを楽譜の絶対時刻でタイムラインに配置。

既定構成:
  F0       PrettyPitch 3話者×3スタイル（checkpoints/f0_3singer.pt、DDPM 10 step）
  音響     LeapSinger 3speaker_gan2d（**v/uv フリー** = use_uv:False, スタイル無し）
  ボコーダ NHVSing v3.2.1（checkpoints/nhv_v3_2_1.onnx）

音響は use_uv で自動的に呼び分ける。v/uv フリーなら infer_mel_uvfree（話者は spk_embed
ベクトルで注入）、v/uv 対応なら infer_mel（uv と style_id が使える）。v/uv 対応の重み
（LeapSinger の配布物）を使いたいときは --acoustic_ckpt で差し替える。
全モデルが hop256/44.1kHz・同一音素セットなので vocab 変換は不要。

PrettyPitch と LeapSinger は共に root の infer.py / train.py を持つため、infer.py は
importlib で別名ロードし、sys.path は PrettyPitch を優先（`train` を PrettyPitch へ解決）。
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

_HERE = Path(__file__).resolve()
_PRETTYPITCH_ROOT = _HERE.parent.parent                      # .../PrettyPitch
_REPO_ROOT = _PRETTYPITCH_ROOT.parent                        # PrettyPitch と LeapSinger を並べて置く親ディレクトリ

# F0 モデル（DDPM 10 step・3 話者×3 スタイル）。リポに同梱（checkpoints/）。
DEFAULT_F0_CKPT = _PRETTYPITCH_ROOT / 'checkpoints/f0_3singer.pt'


def _find_weight(name: str, *dirs: Path) -> Path:
    """dirs を順に見て name（サブフォルダ内も可）が最初に見つかった場所。無ければ dirs[0]/name
    （所在チェックの案内に「置くべき場所」が出るように）。"""
    for d in dirs:
        hits = sorted(Path(d).rglob(name)) if Path(d).is_dir() else []
        if hits:
            return hits[0]
    return Path(dirs[0]) / name


# 音響モデルは LeapSinger の Release（checkpoints/leapsinger/ に zip を展開。
# checkpoints/leapsinger/place_leapsinger_models_here.txt）。ボコーダ nhv_v3_2_1.onnx はリポに同梱。
# 隣に LeapSinger のチェックアウトがある開発環境ではそちらも探す。
_LS_MODELS_DIR = _PRETTYPITCH_ROOT / 'checkpoints/leapsinger'
DEFAULT_ACOUSTIC_CKPT = _find_weight('3speaker_gan2d.pth', _LS_MODELS_DIR,
                                     _REPO_ROOT / 'LeapSinger/checkpoints')
DEFAULT_VOCODER_ONNX = _find_weight('nhv_v3_2_1.onnx', _PRETTYPITCH_ROOT / 'checkpoints',
                                    _LS_MODELS_DIR, _REPO_ROOT / 'LeapSinger/checkpoints')
# LeapSinger の推論コード（<root>/infer.py と leapsinger/ パッケージ）。隣のチェックアウトを既定にする。
DEFAULT_LEAPSINGER_ROOT = _REPO_ROOT / 'LeapSinger'
# 子音長モデル（None にすると全子音が FALLBACK_CONS_DUR の固定長になる）。
# リポに同梱（checkpoints/）。floor は同名の
# duration_model/profiles/cons_dur_3singer.json が自動で対になる。
DEFAULT_CONS_DUR_CKPT = _PRETTYPITCH_ROOT / 'checkpoints/cons_dur_3singer.pt'

def _resolve_phoneset(lang, cons_dur_model, ac_cfg, verbose: bool = True):
    """合成に使う音素表を決める。優先順: --lang > 子音長 ckpt > モジュール既定。

    ここを省くとモジュール既定（日本語）が使われ、英語の子音長 ckpt を渡しても
    全子音がフォールバック長になる。例外も警告も出ないので気づけない。
    音響モデルの音素表と食い違ったら警告する（音素 ID がずれると出力が壊れる）。
    """
    from duration_model import phoneset as _ps_mod
    if lang:
        ps = _ps_mod.load(lang)
    else:
        ps = getattr(cons_dur_model, 'phoneset', None) or _ps_mod.active()
    ac_phs = list(ac_cfg.get('phonemes') or [])
    cur = list(ps.phonemes)
    if ac_phs and cur[:len(ac_phs)] != ac_phs:
        # 前方一致していれば、音素表を末尾へ拡張しただけなので既存 id は不変＝安全。
        warnings.warn(
            f"音素表が食い違っています: 音響モデル {len(ac_phs)} 音素 / "
            f"使用する音素表 '{ps.lang}' {ps.n_phonemes} 音素。"
            f"音素 ID がずれるため合成が壊れます。--lang を確認してください。")
    elif verbose:
        print(f'  音素表: {ps.lang} ({ps.n_phonemes} 音素 / 核 {len(ps.nuclei)})', flush=True)
    return ps


def speaker_names(cons_dur_model) -> dict:
    """話者 id → 名前。子音長モデルの ckpt が持つ spk_map から導出する。

    spk_map（DB名 -> id）は**持続長モデルの ckpt だけ**が記録している（F0・音響の
    ckpt には無い）。そこから逆引きするので、日本語でも英語でも、これから足す
    どの言語でも正しい名前が出る。ハードコードした表は言語を変えると嘘になる。

    同じ id に複数の DB が付く場合（ritsu / ritsu_normal / ritsu_soft はすべて id 2）
    は共通の接頭辞を採る。導出できなければ空 dict を返し、呼び出し側は id だけ出す。
    """
    sm = None
    for obj in (cons_dur_model, getattr(cons_dur_model, 'config', None)):
        if isinstance(obj, dict) and obj.get('spk_map'):
            sm = obj['spk_map']
            break
        sm = getattr(obj, 'spk_map', None) or sm
        if sm:
            break
    if not sm:
        return {}
    byid: dict = {}
    for db, i in sm.items():
        byid.setdefault(int(i), []).append(str(db))
    out = {}
    for i, dbs in byid.items():
        if len(dbs) == 1:
            out[i] = dbs[0]
            continue
        # 共通接頭辞（ritsu_normal / ritsu_soft / ritsu -> ritsu）。無ければ短い方を採る
        pre = dbs[0]
        for d in dbs[1:]:
            while pre and not d.startswith(pre):
                pre = pre[:-1]
        out[i] = pre.rstrip('_') or min(dbs, key=len)
    return out


_SETUP_NOTE = _PRETTYPITCH_ROOT / 'checkpoints/leapsinger/place_leapsinger_models_here.txt'


def _require(path, what: str, hint: str) -> None:
    """無いと分かった時点で「何を、どこに置くか」を書いた FileNotFoundError にする。"""
    if not Path(path).exists():
        raise FileNotFoundError(f'{what} が見つかりません: {path}\n  {hint}\n  （手順: {_SETUP_NOTE}）')


def _load_infer_module(name: str, root: Path):
    """<root>/infer.py を別名 name でロード（root は既に sys.path 上にある前提）。"""
    _require(Path(root) / 'infer.py', 'LeapSinger の推論コード',
             'LeapSinger のリポジトリをこのリポの隣（../LeapSinger）に clone するか、'
             '--leapsinger_root / load_models(leapsinger_root=...) で場所を指定してください')
    spec = importlib.util.spec_from_file_location(name, str(Path(root) / 'infer.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _f0_score_from_seq(seq, spk_id: int = 0, style_id: int = 0) -> dict:
    """PhonemeSequence → synthesize_f0 用スコア。note_midi>0 の音素に midi を付す
    （核音素は必須・子音は baseline 整形。pau は midi なし）。"""
    phonemes = []
    for ph, dur, midi in zip(seq.phonemes, seq.dur_sec, seq.note_midi):
        slot = {'ph': ph, 'dur': float(dur)}
        if int(midi) > 0:
            slot['midi'] = int(midi)
        phonemes.append(slot)
    return {'spk_id': int(spk_id), 'style_id': int(style_id), 'phonemes': phonemes}


_MEL_FRAME_CLIP_THR = 0.23   # 1 フレームの mean exp(ln-mel) の上限
_MEL_LIMIT_ATTACK_MS = 20.0  # 超過フレームの手前から下げ始める幅
_MEL_LIMIT_RELEASE_MS = 250.0  # 下げた後に戻す幅。持続音が沈まないよう長めに取る


def _mel_frame_limit(mel: np.ndarray, frame_rate: float,
                     thr: float = _MEL_FRAME_CLIP_THR) -> np.ndarray:
    """フレーム単位の mel リミッタ。**ボコーダの内部クリップを防ぐのはここだけ。**

    同梱の NHVSing ONNX は出力直前に `[-1, 1]` のクリップを持つ。mel が熱いフレームでは
    そこで波形が潰れ、後段で何をしても取り返せない。だから抑えるならボコーダに入る前、
    つまり mel の段でなければならない。

    各フレーム t の `e[t] = mean_bins exp(mel[:, t])` が thr を超えたら、そのフレームの
    全ビンを `ln(thr / e[t])` だけ一律に下げる。熱いフレームだけを叩くので、通常の音量は
    そのまま残る。下げ量は時間方向に平滑してポンピングとクリックを避ける。

    リリースを 250ms と長めに取るのは、ロングトーンが不自然に沈むのを避けるため。短い
    リリースだと、持続音の途中で下げては戻すを繰り返して音量が波打つ。
    """
    m = np.asarray(mel, np.float64)
    e = np.exp(m).mean(0)                                   # [T]
    d = np.minimum(0.0, np.log(thr / np.maximum(e, 1e-9)))  # [T] <= 0
    if not (d < -1e-4).any():
        return mel
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    atk = max(1, int(round(_MEL_LIMIT_ATTACK_MS * 1e-3 * frame_rate)))
    rel = max(1, int(round(_MEL_LIMIT_RELEASE_MS * 1e-3 * frame_rate)))
    d = minimum_filter1d(d, size=2 * atk + 1, mode='nearest')   # 手前から下げ始める
    d = uniform_filter1d(d, size=2 * rel + 1, mode='nearest')   # ゆっくり戻す
    return (m + d[None, :]).astype(np.asarray(mel).dtype)


_TARGET_MEL_MAG = 0.0738     # 有声フレームの mean exp(ln-mel) の目標。
                             # ボコーダの学習基準 active RMS 0.111 を、実測した変換係数
                             # 1.505（60 フレーズ・3 話者・5 曲の中央値）で割った値。
_MEL_OFFSET_CLAMP = (-1.0, 1.2)   # 学習分布から行き過ぎないための安全域


def _mel_loudness_offset(mel: np.ndarray, uv: np.ndarray) -> float:
    """ボコーダを学習基準の音量で駆動するための、ln-mel への定数オフセット δ。

    **なぜ mel 段で合わせるか。** 波形段で 2 倍すれば、信号と一緒にボコーダのノイズ床も
    2 倍になる。mel 段で上げればボコーダは得意な音量で動き、ノイズは相対的に小さいまま
    出てくる。同じ最終音量でも S/N が違う。

    ln-mel に δ を足すと出力振幅はおよそ exp(δ) 倍になる（実測で確認）。mel magnitude と
    出力 RMS の比は話者と音域で ±1.6dB ばらつくので、ここでは粗く合わせ、残りは波形段の
    小さな残差ゲインで詰める。
    """
    m = np.asarray(mel, np.float64)
    T = min(m.shape[1], len(uv))
    v = np.asarray(uv[:T], bool) if np.asarray(uv).dtype == bool else (np.asarray(uv[:T]) > 0.5)
    seg = m[:, :T][:, v] if v.any() else m[:, :T]
    mag = float(np.exp(seg).mean())
    if mag < 1e-6:
        return 0.0
    d = float(np.log(_TARGET_MEL_MAG / mag))
    return float(min(max(d, _MEL_OFFSET_CLAMP[0]), _MEL_OFFSET_CLAMP[1]))


def _vocode(ctx, mel: np.ndarray, logf0: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """mel → 波形。**LeapSinger の `mel_to_wav` と同じ入力仕様だが、正規化はしない。**

    `mel_to_wav` は返す直前にフレーズをピーク 0.95 へ正規化する。単体で 1 フレーズを
    鳴らすには妥当だが、曲を組み立てるとフレーズごとの音量がいちばん大きい 1 サンプルで
    決まってしまう。破裂音が 1 つ入ればそのフレーズが沈み、静かなフレーズは持ち上がる。
    音響モデルの出力自体は揃っている（実測でフレーズ間の mel 平均の差は 0.6dB）ので、
    ここでは生のまま受け取り、正規化は曲全体に 1 度だけ掛ける（`_normalize`）。
    """
    mel = np.asarray(mel) + _mel_loudness_offset(mel, uv)   # 粗いラウドネス合わせ
    mel = _mel_frame_limit(mel, float(ctx.frame_rate))     # そのうえで熱いフレームを叩く
    m = np.ascontiguousarray(np.asarray(mel).T[None], dtype=np.float32)      # [1,T,128]
    T = m.shape[1]

    def _fit(a):
        a = np.asarray(a, dtype=np.float32).reshape(-1)[:T]
        return np.pad(a, (0, max(0, T - len(a))))

    f0hz = (2.0 ** _fit(logf0))[None, None].astype(np.float32)               # [1,1,T]
    uv_in = (1.0 - _fit(uv))[None, None].astype(np.float32)                  # 1 = 無声
    out = ctx.voc.session.run(["waveform"], {"mel": m, "f0": f0hz, "uv": uv_in})[0]
    return np.asarray(out, np.float32).reshape(-1)


_ACTIVE_THRESH_DB = -40.0      # これを超える振幅のサンプルだけで RMS を測る（無音を除く）

def _active_rms(y: np.ndarray, thresh_db: float = _ACTIVE_THRESH_DB) -> float:
    """NHVSing の preprocess.active_rms と同じ定義。無音を除いた実効値。"""
    a = np.asarray(y, np.float64)
    t = 10.0 ** (thresh_db / 20.0)
    loud = a[np.abs(a) > t]
    if len(loud) < max(1, int(len(a) * 0.01)):
        return float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0
    return float(np.sqrt(np.mean(loud ** 2)))


def _normalize(wav: np.ndarray, sr: int, voiced: np.ndarray | None, mode: str) -> np.ndarray:
    """曲全体の音量。既定では**何もしません**。

    音量は mel の段で決まります（`_mel_loudness_offset` がボコーダの学習基準に合わせる）。
    そこで合わせたものを波形側で掛け直すと、せっかく合わせた駆動音量の意味が薄れるうえ、
    2 か所で音量を決めることになって追いにくくなります。実測でも mel 段だけで曲・話者を
    またいだ振れ幅は 1.55 dB、ピークは 0.32〜0.68 に収まり、1.0 超えはありませんでした。

    'none'（既定） 何もしない。音量は mel 段で決まったまま。
    'peak'         ピークを 0.95 へ。曲の中でいちばん大きい 1 サンプルで全体が決まる。
                   フレーズ間の相対音量が崩れるので、旧挙動を再現したいときだけ。
    """
    w = np.asarray(wav, np.float32)
    if mode == 'none' or w.size == 0:
        return w
    if mode == 'peak':
        pk = float(np.abs(w).max())
        return (w / pk * 0.95).astype(np.float32) if pk > 1e-6 else w
    raise ValueError(f"normalize は none / peak のいずれか: {mode!r}")


def _align_durs(ph_durs: np.ndarray, target_T: int) -> np.ndarray:
    """ph_durs の総和を target_T に合わせる（末尾音素で ±調整）。各要素は 1 以上を保つ。"""
    ph_durs = np.asarray(ph_durs, dtype=np.int64).copy()
    diff = int(target_T - int(ph_durs.sum()))
    if diff != 0 and len(ph_durs) > 0:
        ph_durs[-1] = max(1, ph_durs[-1] + diff)
    return ph_durs


def _speaker_vector(ac_model, spk_id: int, device: str):
    """多話者モデルの離散話者 lookup と同値の hidden 次元ベクトル。

    infer_mel_uvfree は spk_embed を渡さないと spk_id を一切モデルへ送らない
    （kw に spk_id が入らず _spk_add が cond を素通しする）ので、3話者モデルでも
    話者条件が効かなくなる。ここで同じ値を明示的に作って渡す。
    """
    import torch
    bank = getattr(ac_model, 'spk_bank', None)
    proj = getattr(ac_model, 'spk_proj', None)
    if bank is None or proj is None:
        return None
    with torch.no_grad():
        idx = torch.tensor([int(spk_id)], device=torch.device(device))
        return proj(bank(idx))[0].detach().cpu().numpy()


@dataclass
class SvsContext:
    """ロード済みの 3 モデルと派生設定（notebook から再利用するための束）。"""
    pp: Any                 # PrettyPitch infer モジュール
    ls: Any                 # LeapSinger infer モジュール
    f0_model: Any
    f0_cfg: dict
    ac_model: Any
    ac_cfg: dict
    voc: Any
    ph2id: dict
    frame_rate: float
    sample_rate: int
    hop: int
    spk_id: int
    spk_vec: Optional[np.ndarray]
    device: str
    cons_dur_model: Any = None          # 子音長モデル（None = 固定長フォールバック）
    cons_dur_profile: Any = None        # floor プロファイル
    speaker_names: dict = field(default_factory=dict)   # id -> 名前（ckpt 由来・表示用）
    phoneset: Any = None                # 音素表。既定 = 子音長 ckpt が記録したもの

    @property
    def use_uv(self) -> bool:
        return bool(self.ac_cfg.get('use_uv', True))


def load_models(
    f0_ckpt=DEFAULT_F0_CKPT,
    acoustic_ckpt=DEFAULT_ACOUSTIC_CKPT,
    vocoder_onnx=DEFAULT_VOCODER_ONNX,
    leapsinger_root=DEFAULT_LEAPSINGER_ROOT,
    spk_id: int = 0,
    device: str = 'cpu',
    cons_dur_ckpt=DEFAULT_CONS_DUR_CKPT,
    lang: Optional[str] = None,
    verbose: bool = True,
) -> SvsContext:
    """F0・音響・ボコーダ・子音長モデルをまとめてロードして SvsContext を返す。

    cons_dur_ckpt=None、または見つからない場合は子音長を固定値にフォールバックする。
    """
    pp_root = str(_PRETTYPITCH_ROOT)
    ls_root = str(Path(leapsinger_root).resolve())
    # `train` を PrettyPitch へ解決するため PrettyPitch を先頭に、LeapSinger をその次に。
    for r in (ls_root, pp_root):
        while r in sys.path:
            sys.path.remove(r)
    sys.path.insert(0, ls_root)
    sys.path.insert(0, pp_root)

    pp = _load_infer_module('pp_infer', _PRETTYPITCH_ROOT)      # prettypitch.* / train(PrettyPitch)
    ls = _load_infer_module('ls_infer', Path(ls_root))          # leapsinger.*（自己完結）

    if verbose:
        print('loading F0 model ...', flush=True)
    f0_model, f0_cfg = pp.load_model(str(f0_ckpt), device)
    if verbose:
        print('loading acoustic model ...', flush=True)
    _require(acoustic_ckpt, 'LeapSinger の音響モデル',
             'python tools/download_leapsinger.py で取得できます'
             '（既に持っているなら --acoustic_ckpt で場所を指定してください）')
    ac_model, ac_cfg = ls.load_acoustic(str(acoustic_ckpt), device)
    if verbose:
        print('loading vocoder ...', flush=True)
    _require(vocoder_onnx, 'ボコーダ nhv_v3_2_1.onnx',
             'checkpoints/nhv_v3_2_1.onnx はリポジトリに同梱されています。消した場合は '
             'NHVSing の Release から置き直すか、--vocoder_onnx で場所を指定してください')
    voc = ls.load_vocoder(str(vocoder_onnx), sr=int(ac_cfg.get('sample_rate', 44100)))

    cons_model = cons_prof = None
    if cons_dur_ckpt is not None and Path(cons_dur_ckpt).exists():
        from svs.cons_dur import load_cons_dur
        cons_model, cons_prof = load_cons_dur(cons_dur_ckpt, device=device)
        if verbose:
            print(f'  cons-dur: {Path(cons_dur_ckpt).name} '
                  f'(floor {"プロファイル" if cons_prof.cons_floor.get("k", 0.01) != 0.01 else "組込み既定"})',
                  flush=True)
    elif verbose:
        print('  cons-dur: なし → 子音長は固定値フォールバック', flush=True)

    ctx = SvsContext(
        pp=pp, ls=ls, f0_model=f0_model, f0_cfg=f0_cfg,
        ac_model=ac_model, ac_cfg=ac_cfg, voc=voc,
        ph2id={p: i for i, p in enumerate(ac_cfg['phonemes'])},
        frame_rate=float(f0_cfg['frame_rate']),
        sample_rate=int(f0_cfg['sample_rate']),
        hop=int(f0_cfg.get('hop', 256)),
        spk_id=int(spk_id),
        spk_vec=_speaker_vector(ac_model, spk_id, device),
        device=device,
        cons_dur_model=cons_model,
        cons_dur_profile=cons_prof,
        speaker_names=speaker_names(cons_model),
        phoneset=_resolve_phoneset(lang, cons_model, ac_cfg, verbose),
    )
    if verbose:
        _nm = ctx.speaker_names.get(int(spk_id))
        print(f'  acoustic: use_uv={ctx.use_uv} n_speakers={ac_cfg.get("n_speakers")} '
              f'n_styles={ac_cfg.get("n_styles")}  | spk_id={spk_id}'
              + (f' ({_nm})' if _nm else ''), flush=True)
    return ctx


def render_phrase(ctx: SvsContext, phrase, *, num_steps: Optional[int] = None, seed: int = 0,
                  plain_scale: float = 1.0, vib_scale: float = 1.0,
                  style_id: int = 0, style_mix=None, f0_only: bool = False) -> dict:
    """1 フレーズを合成し、途中経過ごと返す。

    num_steps: 音響モデルのサンプリング歩数。触らないでください。同梱モデルは 1 歩で出るよう
      学習した rectified flow で、ckpt も 1 を指定しています。増やすと遅くなるうえ音も悪く
      なります。
    style_mix: [n_styles] の混合比（連続スタイル）。F0 モデルと子音長モデルのスタイル埋め込みを
      この比で混ぜる。floor プロファイルと音響モデルは style_id のまま。
    f0_only: True なら持続長と F0 まで（音響・ボコーダを走らせない）。'mel' / 'wav' は None。

    戻り値: {'seq', 'ph_durs', 'f0', 'mel', 'wav', 'start_sec', 'onset0_sec'}
      start_sec  : この波形を置きたい絶対時刻 [s]。**負になりうる**（先頭 onset を
                   楽譜のノート開始より前に置くため）。0 でクランプせず、render_score が
                   全フレーズを見てから一括でずらす。
      onset0_sec : 先頭 onset クラスタの長さ [s]（ノート開始より前倒しされる量）。
      seq  = PhonemeSequence、f0 = synthesize_f0 の dict、mel = [n_mels, T]
    """
    from svs.duration_postprocess import (postprocess_durations, to_frames,
                                          split_phrase_syllables, split_syllable)

    if num_steps is None:
        num_steps = int(ctx.ac_cfg.get('infer_steps', 1))

    # 音素表は ctx.phoneset（既定 = 子音長 ckpt が記録したもの）を全段へ明示的に渡す。
    # 渡さないとモジュール既定（日本語）に落ち、英語 ckpt でも全子音がフォールバック長に
    # なる（例外も警告も出ない）。
    ps = ctx.phoneset

    # 多モーラノートをグループへ展開してから子音長を予測する（内部子音も予測対象にするため）。
    # 展開は冪等なので、モデルが無い場合もそのまま通してよい。
    phrase = split_phrase_syllables(phrase, ps)
    if ctx.cons_dur_model is not None:
        from svs.cons_dur import predict_cons_naturals
        on_nat, cd_nat, co_nat = predict_cons_naturals(
            phrase, ctx.cons_dur_model, spk_id=ctx.spk_id, style_id=style_id,
            device=ctx.device, phoneset=ps, with_core=True, style_mix=style_mix)
        # floor も話者別に引く（profile.per_speaker があれば。無ければ全話者共通値）。
        # モデル側は spk_emb で話者を見ているので、floor だけ共通だと話者差がここで潰れる。
        seq = postprocess_durations(phrase, ctx.cons_dur_profile, on_nat, cd_nat,
                                    core_nat=co_nat, rest_before=1e9, phoneset=ps,
                                    spk_id=ctx.spk_id, style_id=style_id)
    else:
        seq = postprocess_durations(phrase, rest_before=1e9, phoneset=ps)
    ph_durs, _T = to_frames(seq.dur_sec, ctx.frame_rate)

    # ── F0（2段ピッチ補正込み）
    f0o = ctx.pp.synthesize_f0(
        ctx.f0_model, ctx.f0_cfg,
        _f0_score_from_seq(seq, spk_id=ctx.spk_id, style_id=style_id),
        device=ctx.device, seed=seed,
        plain_scale=plain_scale, vib_scale=vib_scale, style_mix=style_mix)
    f0_hz = np.asarray(f0o['f0_hz'], np.float32)
    uv = np.asarray(f0o['uv'], np.float32)
    Tf = len(f0_hz)
    ph_durs = _align_durs(ph_durs, Tf)
    logf0 = np.log2(np.maximum(f0_hz, 1.0)).astype(np.float32)   # gap-less（ボコーダ用）

    ph_ids = np.asarray([ctx.ph2id[p] for p in seq.phonemes], np.int64)

    if f0_only:
        return {'seq': seq, 'ph_durs': ph_durs, 'f0': f0o, 'mel': None, 'wav': None,
                'start_sec': None, 'onset0_sec': None}

    # ── 音響 mel（モデルの use_uv で呼び分け）
    if ctx.use_uv:
        item = dict(ph_ids=ph_ids, ph_durs=ph_durs.astype(np.int64),
                    f0_logf0=logf0, uv=uv,
                    spk_id=int(ctx.spk_id), style_id=int(style_id))
        mel = ctx.ls.infer_mel(ctx.ac_model, item, num_steps=num_steps,
                               device=ctx.device, seed=seed)
    else:
        # uvfree は f0_hz（logf0 ではない）を取る。話者は spk_embed で明示注入。
        mel = ctx.ls.infer_mel_uvfree(
            ctx.ac_model, ph_ids, ph_durs.astype(np.int64), f0_hz,
            num_steps=num_steps, spk_embed=ctx.spk_vec,
            device=ctx.device, seed=seed)

    # ── ボコーダ（mel→wav, NHV ONNX）
    # uv は **uv_hard = (F0 モデルの vuv > 0.5) & (pau でない)** を渡す（ボコーダの入力仕様）。
    frame_ph = np.repeat(ph_ids, np.maximum(ph_durs.astype(np.int64), 0))[:Tf]
    if len(frame_ph) < Tf:
        frame_ph = np.pad(frame_ph, (0, Tf - len(frame_ph)), constant_values=ctx.ph2id.get('pau', 0))
    is_pau_fr = np.isin(frame_ph, [ctx.ph2id[p] for p in ('pau',) if p in ctx.ph2id])
    uv_hard = ((uv > 0.5) & ~is_pau_fr).astype(np.float32)
    wav = _vocode(ctx, mel, logf0, uv_hard)

    # ── 配置：フレーズ先頭ノートの**音節核**が楽譜の時刻に来るようにする。
    # 波形は [leading pau][先頭 onset][核 ...] の順なので、その 2 つを差し引いた時刻へ置く。
    # 差し引かないと、フレーズ全体が先頭 onset の長さだけ遅れる。ずれ量は先頭子音に
    # よって変わる（日本語で中央値 35ms・最大 273ms、英語で中央値 110ms・最大 470ms）ため、
    # 全体を一律にずらしても直らない＝フレーズ間の相対位置が崩れる。
    # フレーズ内部の onset は前のノートへ前倒しされており、ここはそれと同じ規約にあたる。
    n_on0 = len(split_syllable(phrase.notes[0].phonemes, ps)[0]) if phrase.notes else 0
    onset0_sec = float(seq.dur_sec[1:1 + n_on0].sum()) if n_on0 else 0.0
    lead_sec = float(seq.dur_sec[0]) if len(seq.dur_sec) else 0.0
    start_sec = phrase.start_sec - lead_sec - onset0_sec
    return {'seq': seq, 'ph_durs': ph_durs, 'f0': f0o, 'mel': mel,
            'wav': wav, 'start_sec': start_sec, 'onset0_sec': onset0_sec}


def render_score(
    score_path,
    f0_ckpt=DEFAULT_F0_CKPT,
    acoustic_ckpt=DEFAULT_ACOUSTIC_CKPT,
    vocoder_onnx=DEFAULT_VOCODER_ONNX,
    leapsinger_root=DEFAULT_LEAPSINGER_ROOT,
    spk_id: int = 0,
    num_steps: Optional[int] = None,
    device: str = 'cpu',
    seed: int = 0,
    mora_table=None,
    style_id: int = 0,
    plain_scale: float = 1.0,
    vib_scale: float = 1.0,
    cons_dur_ckpt=DEFAULT_CONS_DUR_CKPT,
    lang: Optional[str] = None,
    ctx: Optional[SvsContext] = None,
    transpose: int = 0,
    normalize: str = 'none',
    part=None,
):
    """Render a score file (.ust / .musicxml / .mxl) to audio.

    Args:
        score_path: the score. It is split into phrases at rests of 0.5 s or longer.
        f0_ckpt, acoustic_ckpt, vocoder_onnx, leapsinger_root, cons_dur_ckpt: model
            locations; the defaults are the bundled files and the sibling LeapSinger
            checkout (see checkpoints/leapsinger/place_leapsinger_models_here.txt).
            ``cons_dur_ckpt=None`` falls back to a fixed consonant length.
        spk_id, style_id: speaker and style ids (names: ``SvsContext.speaker_names``).
        num_steps: acoustic-model sampling steps. Leave it alone. The bundled model is a
            rectified flow trained to go in one step, and its checkpoint asks for one.
            More steps is not better here — it is slower and it sounds worse.
        seed: fixes the F0 and acoustic sampling noise.
        mora_table: lyric → phoneme table (default: the language pack's).
        plain_scale, vib_scale: expression and vibrato strength (1.0 = as trained).
        lang: language pack name; default: the one recorded in the duration checkpoint.
        ctx: a pre-loaded ``SvsContext`` from ``load_models`` (skips loading).
        transpose: shift every note by this many semitones, to move a score into a
            speaker's range. The models are trained on each speaker's own register, so
            a score far outside it sounds strained.
        normalize: what to do to the waveform's level. The default ``'none'`` does
            nothing: the level is already set on the mel, where the vocoder is driven at
            the level it was trained on. ``'peak'`` scales the peak to 0.95, the old
            behaviour, which lets one loud sample decide the level of everything else.
        part: which MusicXML part to sing, by part id or part-name. The default picks the
            first part that has lyrics, so a score with an accompaniment works as it is.

    Returns:
        ``(wav, sample_rate)`` — float32 mono audio with the phrases placed on the
        score timeline (0.25 s of silence at both ends of every phrase).
    """
    if ctx is None:
        ctx = load_models(f0_ckpt=f0_ckpt, acoustic_ckpt=acoustic_ckpt,
                          vocoder_onnx=vocoder_onnx, leapsinger_root=leapsinger_root,
                          spk_id=spk_id, device=device, cons_dur_ckpt=cons_dur_ckpt,
                          lang=lang)

    from svs.score_io import load_score
    phrases = load_score(score_path, mora_table=mora_table, phoneset=ctx.phoneset,
                         part=part)
    if transpose:
        for phr in phrases:
            for n in phr.notes:
                n.pitch_midi = int(n.pitch_midi) + int(transpose)
        print(f'transposed by {transpose:+d} semitones', flush=True)
    print(f'{len(phrases)} phrases; rendering ...', flush=True)

    rendered = []   # (start_sec, wav)
    for k, phr in enumerate(phrases):
        r = render_phrase(ctx, phr, num_steps=num_steps, seed=seed,
                          plain_scale=plain_scale, vib_scale=vib_scale, style_id=style_id)
        # 有声マスクもフレーム→サンプルへ広げて持っておく（正規化を有声区間で測るため）
        _uv = np.asarray(r['f0']['uv'], np.float32) > 0.5
        _hop = max(1, int(round(ctx.sample_rate / ctx.frame_rate)))
        _uvs = np.repeat(_uv, _hop)[:len(r['wav'])]
        if len(_uvs) < len(r['wav']):
            _uvs = np.pad(_uvs, (0, len(r['wav']) - len(_uvs)))
        rendered.append((r['start_sec'], r['wav'], _uvs))
        print(f"  phrase {k}: {len(r['seq'].phonemes)} ph, {len(r['f0']['f0_hz'])} frames, "
              f"{len(r['wav'])/ctx.sample_rate:.2f}s @ {phr.start_sec:.2f}s "
              f"(onset {r['onset0_sec']*1000:.0f}ms 前倒し)", flush=True)

    # 置きたい時刻が負になるフレーズがあれば、**曲全体**をその分だけ後ろへずらす。
    # フレーズ個別に 0 でクランプすると、曲頭のフレーズだけ遅れてフレーズ間の相対位置が
    # 崩れる（それを避けるためにこの配置にしている）。増えるのは出力冒頭の無音だけ。
    offset = max(0.0, -min((t for t, _, _ in rendered), default=0.0))
    if offset > 0:
        print(f'  先頭に {offset*1000:.0f}ms の無音を足して曲全体をずらします', flush=True)
    placed = [(max(0, int(round((t + offset) * ctx.sample_rate))), w, m)
              for t, w, m in rendered]

    total = max((s + len(w) for s, w, _ in placed), default=0)
    out = np.zeros(total, np.float32)
    voiced = np.zeros(total, bool)
    for s, w, m in placed:
        out[s:s + len(w)] += w
        voiced[s:s + len(m)] |= m
    return _normalize(out, ctx.sample_rate, voiced, normalize), ctx.sample_rate


def _write_wav(path, wav: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(path, wav, sr)
        return
    except Exception:
        pass
    from scipy.io import wavfile
    wavfile.write(path, sr, (np.clip(wav, -1, 1) * 32767).astype(np.int16))


def main():
    ap = argparse.ArgumentParser(description='SVS レンダリング（楽譜→wav）')
    ap.add_argument('score', help='楽譜ファイル（.ust / .musicxml / .mxl）')
    ap.add_argument('-o', '--out', default='svs_render.wav', help='出力 wav')
    ap.add_argument('--f0_ckpt', default=str(DEFAULT_F0_CKPT))
    ap.add_argument('--acoustic_ckpt', default=str(DEFAULT_ACOUSTIC_CKPT),
                    help='音響 ckpt（既定 = v/uv フリー。v/uv 対応の重みもそのまま差せる）')
    ap.add_argument('--vocoder_onnx', default=str(DEFAULT_VOCODER_ONNX))
    ap.add_argument('--leapsinger_root', default=str(DEFAULT_LEAPSINGER_ROOT))
    ap.add_argument('--mora_table', default=None,
                    help='歌詞→音素のモーラ表（既定 dict/ja.mora）。他言語は差し替える')
    ap.add_argument('--spk_id', type=int, default=0, help='話者 0=oniku 1=natsume 2=ritsu')
    ap.add_argument('--style_id', type=int, default=0,
                    help='スタイル（F0 は 3 スタイル、音響は v/uv 版のみ有効）')
    ap.add_argument('--plain_scale', type=float, default=1.0, help='抑揚の強さ')
    ap.add_argument('--vib_scale', type=float, default=1.0, help='ビブラートの強さ')
    ap.add_argument('--part', default=None,
                    help='MusicXML で歌うパート（id か part-name）。既定は歌詞のある最初のパート')
    ap.add_argument('--transpose', type=int, default=0,
                    help='移調（半音）。楽譜を話者の音域へ寄せる')
    ap.add_argument('--lang', default=None,
                    help='言語パック（dict/<lang>.yaml）。既定は子音長 ckpt が記録した言語')
    ap.add_argument('--cons_dur_ckpt', default=str(DEFAULT_CONS_DUR_CKPT),
                    help="子音長モデル。'none' で固定長フォールバック（A/B 用）")
    ap.add_argument('--normalize', choices=('none', 'peak'), default='none',
                    help='波形段の音量調整。既定 none（音量は mel 段で決まる）/ '
                         'peak=ピーク 0.95（旧挙動）')
    ap.add_argument('--num_steps', type=int, default=None,
                    help='音響モデルのサンプリング歩数。既定は ckpt の infer_steps（＝1）。'
                         '同梱モデルは 1 歩で出るよう学習した rectified flow なので、'
                         '増やしても良くなりません（遅くなり音も悪くなります）')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    wav, sr = render_score(
        a.score, f0_ckpt=a.f0_ckpt, acoustic_ckpt=a.acoustic_ckpt,
        vocoder_onnx=a.vocoder_onnx, leapsinger_root=a.leapsinger_root,
        spk_id=a.spk_id, num_steps=a.num_steps, device=a.device, seed=a.seed,
        mora_table=a.mora_table, style_id=a.style_id,
        plain_scale=a.plain_scale, vib_scale=a.vib_scale,
        cons_dur_ckpt=(None if str(a.cons_dur_ckpt).lower() == 'none' else a.cons_dur_ckpt),
        lang=a.lang, transpose=a.transpose, normalize=a.normalize, part=a.part)
    _write_wav(a.out, wav, sr)
    print(f'wrote {a.out}  ({len(wav)/sr:.2f}s, {sr} Hz)')


if __name__ == '__main__':
    main()
