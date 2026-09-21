"""
phrase_cut.py

lab の静音区間(pau/br)を基準に音声をフレーズ単位で分割し、フレーズ単位の .npz を書き出す。
LeapSinger の同名モジュールの mel-free 版: PrettyPitch は F0 モデルなので mel は保存せず、
無音句の除去(音量ゲート)はフレーム RMS ラウドネス(dBFS)で行う。

■ フレーズ分割アルゴリズム(LeapSinger と同一):
  静音区間 = pau/br のみ(cl/N/母音/子音 = 有音)。
  1. lab_segs を silence/voiced の極大 run に統合。
  2. 長さ ≥ split_silence_sec(既定0.5s)の静音 = CUT。CUT でフレーズグループを分ける。
  3. グループ内の content = 有音 + 内部の短い静音(<split)。内部 pau/br は句内に残す
     — トラック全体の F0 を生成できるよう、静音区間の処理を学ばせるため。
  4. content 長が max_dur_sec(既定10s)を超えたら、10s を越えた直後の ≥min_silence_sec の
     内部静音で追加分割(再帰)。≥min の静音が無ければ上限なしで長いまま(仕様)。
  5. 各 content の端に実際の静音を付与: lead/trail = min(隣接静音の実長, edge_max_sec=0.5)。
     直前静音が <lead_min_sec(0.1s)なら不足分を人工静音+pau で補う。
  6. 音量ゲート: 句のフレーム RMS の p90 < min_loudness_db(既定 -50 dBFS)なら保存しない
     (=ラベル↔音声ズレ由来の無音句を除去)。
  7. 境界を hop_size 倍数にスナップ、f0/uv/phoneme をスライス、dur は frame から再構成。

■ 出力 npz フィールド(フレーズ単位):
    phoneme_ids : int16   [L_ph]  phoneme vocab ID
    dur_sec     : float32 [L_ph]  phoneme duration [s]
    f0_interp   : float32 [T]     interpolated F0 [Hz]。名実ともにギャップレス:
                                  人工静音パディングも端の値でフラット延長する
                                  (f0=0 を混ぜると残差教師に -baseline の外れ値が出るため。
                                   無声であることは uv が持つ)
    uv          : float32 [T]     voiced mask (1=voiced)。人工静音パディングは 0
音高に関する派生情報(核音素の代表値・baseline)は一切保存しない — dataset.py が導出する。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── 定数(LeapSinger と同値。min_loudness のみ RMS dBFS 尺度に置換) ─────────────

# 真の静音。母音/子音は有音。**言語パック（dict/<lang>.yaml）から引く**ので直書きしない。
# load_lab が先に sil_map で pau へ畳んでいるため、実際に現れるのは pau だけだが、
# 生の記号が渡された場合にも備えて sil_map のキーも含めておく。
def _silence_phones() -> frozenset:
    try:
        from duration_model import phoneset as _ps_mod
        ps = _ps_mod.active()
        return frozenset({ps.pau}) | frozenset(ps.sil_map)
    except Exception:                                   # noqa: BLE001
        return frozenset({'pau', 'sil', 'br', 'sp', 'sli'})


_MIN_SILENCE_SEC   = 0.1    # フレーズ境界候補・max_dur 追加分割点の最小静音長
_SPLIT_SILENCE_SEC = 0.5    # これ以上の静音で必ず区切る(CUT)。未満は句内に残す
_EDGE_MAX_SEC      = 0.5    # 句の端に含める静音の上限(実長を使い最大0.5s)
_LEAD_MIN_SEC      = 0.1    # 句頭に最低確保する静音(不足は人工静音で補う)
_MAX_DUR_SEC       = 10.0   # これを超えたら次の ≥min 静音で追加分割(ハード上限なし)
_MIN_LOUDNESS_DB   = -50.0  # フレームRMS dBFS の p90 がこれ未満なら無音句として除去
# 促音(cl)固有の規則: これより長い cl は誤ラベル休符とみなし静音扱い(短い cl は有音)。
# 'cl' は日本語パック固有の音素なので、cl を持たない言語では発火しない（無害）。
_CL_PAU_SEC        = 0.3


def _is_silence(ph: str, dur: float, silence_phones, cl_pau_sec: float) -> bool:
    """静音判定。pau/br/sil… は静音。cl は cl_pau_sec より長ければ静音(誤ラベル休符)扱い。"""
    if ph in silence_phones:
        return True
    if cl_pau_sec > 0 and ph == 'cl' and dur > cl_pau_sec:
        return True
    return False


def frame_loudness_db(clip: np.ndarray, hop: int, n_frames: int) -> np.ndarray:
    """フレーム毎 RMS ラウドネス [dBFS] float32[n_frames]。無音句ゲート用(mel の代役)。"""
    need = n_frames * hop
    x = clip[:need]
    if len(x) < need:
        x = np.pad(x, (0, need - len(x)))
    rms = np.sqrt(np.mean(x.reshape(n_frames, hop) ** 2, axis=1))
    return (20.0 * np.log10(np.maximum(rms, 1e-6))).astype(np.float32)


# ── 静音区間の無音化(息継ぎ対策) ──────────────────────────────────────────────
# ラベル上 pau なのに実際は息継ぎが入っており、F0 抽出器がそれを有声と判定して
# 歌唱より高い F0 を返す。natsume(男声)では pau フレームの 35% が有声判定・その F0
# 中央値 267Hz に対し核母音は 196Hz と逆転していた。F0 抽出の前に音声を消す。
# 静音の定義は _is_silence と同一(pau + 長い cl)なので、フレーズ分割と judgement がずれない。

def build_sil_mask(lab_segs: list, total_frames: int, frame_rate: float, *,
                   silence_phones=None, cl_pau_sec: float = _CL_PAU_SEC) -> np.ndarray:
    """ラベル基準の静音(pau / 長い cl)フレームマスク bool[T]。"""
    sph = silence_phones or _silence_phones()
    sil = np.zeros(int(total_frames), dtype=bool)
    for s, e, ph in lab_segs:
        if _is_silence(ph, e - s, sph, cl_pau_sec):
            a = max(0, int(round(s * frame_rate)))
            b = min(int(total_frames), int(round(e * frame_rate)))
            if b > a:
                sil[a:b] = True
    return sil


def clip_pau_gain(sil_clip: np.ndarray, fade_frames: int) -> np.ndarray:
    """1 フレーズ内の静音化ゲイン g[n](float32)。距離はクリップ内の最近傍有音まで測るので、
    フレーズ端の静音は片側傾斜(クリップ端で必ず 0)、内部の静音だけ両側傾斜になる。
    端の静音 run が fade_frames より短いときは傾斜を run 長へ圧縮し、端点で 0 に到達させる。

    両端を残すのは意図的: 尾曳き(直前の音の減衰)と歌い出し(次の音の立ち上がり)は
    歌唱の一部なので消さない。潰すのは中央の core(g≈0)だけ。
    run が fade の 2 倍より短ければ core は生じない = 短い pau は潰れない。"""
    n = len(sil_clip)
    if not sil_clip.any():
        return np.ones(n, dtype=np.float32)
    if fade_frames <= 0 or sil_clip.all():
        return (~sil_clip).astype(np.float32)
    vi = np.flatnonzero(~sil_clip)
    d = np.abs(np.arange(n)[:, None] - vi[None, :]).min(axis=1)
    g = np.clip(1.0 - d / fade_frames, 0.0, 1.0).astype(np.float32)
    lead = int(vi[0])                      # 先頭 sil run 長
    if lead > 0:
        L = min(fade_frames, lead)
        dd = lead - np.arange(lead)        # 有音境界までの距離(端で最大)
        g[:lead] = np.minimum(g[:lead], np.clip(1.0 - dd / L, 0.0, 1.0))
    trail = n - 1 - int(vi[-1])            # 末尾 sil run 長
    if trail > 0:
        L = min(fade_frames, trail)
        dd = np.arange(1, trail + 1)       # 有音境界からの距離(端で最大=trail)
        g[n - trail:] = np.minimum(g[n - trail:], np.clip(1.0 - dd / L, 0.0, 1.0))
    return g


def gain_to_samples(g: np.ndarray, hop: int, n_samples: int) -> np.ndarray:
    """フレーム毎ゲイン g[n] をサンプル毎へ線形補間する float32[n_samples]。
    np.repeat だと 1 フレーム毎に ~1dB の段差が立ちクリックになるので補間する。"""
    n = len(g)
    if n == 0:
        return np.ones(n_samples, dtype=np.float32)
    if n == 1:
        return np.full(n_samples, float(g[0]), dtype=np.float32)
    x = np.arange(n_samples, dtype=np.float64) / float(hop)
    return np.interp(x, np.arange(n, dtype=np.float64), g).astype(np.float32)


# ── run 統合 ─────────────────────────────────────────────────────────────────

def _merge_runs(lab_segs: list, silence_phones=None, cl_pau_sec: float = _CL_PAU_SEC) -> list:
    """連続する同種(静音/有音)を統合し (is_sil, start_sec, end_sec) の極大 run を返す。"""
    runs: list = []
    for s, e, ph in lab_segs:
        if e <= s:
            continue
        is_sil = _is_silence(ph, e - s, silence_phones or _silence_phones(), cl_pau_sec)
        if runs and runs[-1][0] == is_sil:
            runs[-1] = (is_sil, runs[-1][1], e)
        else:
            runs.append((is_sil, s, e))
    return runs


# ── フレーズ境界検出 ────────────────────────────────────────────────────────

def find_phrase_spans(
    lab_segs: list,
    total_sec: float,
    *,
    silence_phones = None,
    min_silence_sec:   float = _MIN_SILENCE_SEC,
    split_silence_sec: float = _SPLIT_SILENCE_SEC,
    edge_max_sec:      float = _EDGE_MAX_SEC,
    lead_min_sec:      float = _LEAD_MIN_SEC,
    max_dur_sec:       float = _MAX_DUR_SEC,
    cl_pau_sec:        float = _CL_PAU_SEC,
    hop_size:          int   = 256,
    sr:                int   = 44_100,
) -> List[Tuple[int, int, int]]:
    """フレーズ区間 [(start_frame, end_frame, n_art_lead, n_art_trail), ...] を返す。
    境界は hop_size 倍数(フレーム境界)にスナップ済み。n_art_lead/n_art_trail>0 のとき
    その数だけ端に人工静音フレームを付与する(cut_phrases 側で処理)。"""
    runs = _merge_runs(lab_segs, silence_phones or _silence_phones(), cl_pau_sec)
    if not runs:
        return []
    frame_rate = sr / hop_size
    total_frames = int(round(total_sec * frame_rate))

    # ── content(有音-有音, CUT静音で区切る)を抽出 ──
    contents: list = []   # (c_s, c_e)
    cur_s = None; cur_e = None
    for is_sil, s, e in runs:
        if is_sil and (e - s) >= split_silence_sec:      # CUT: 現 content を閉じる
            if cur_s is not None:
                contents.append((cur_s, cur_e)); cur_s = None
        elif not is_sil:                                  # 有音: content を伸ばす(末尾=最後の有音)
            if cur_s is None:
                cur_s = s
            cur_e = e
        # 短い静音(<split): content 内なら [cur_s,cur_e] に自然に含まれる。leading/trailing は無視
    if cur_s is not None:
        contents.append((cur_s, cur_e))
    if not contents:
        return []

    # ── max_dur 超えを 10s 直後の内部静音で追加分割(再帰) ──
    def _subsplit(c_s, c_e) -> list:
        if c_e - c_s <= max_dur_sec:
            return [(c_s, c_e)]
        cand = [(s, e) for is_sil, s, e in runs
                if is_sil and (e - s) >= min_silence_sec and s > c_s + max_dur_sec and e <= c_e]
        if not cand:
            return [(c_s, c_e)]                           # 切れる静音が無ければ長いまま(上限なし)
        s0, e0 = cand[0]                                  # 10s を越えた最初の内部静音
        return _subsplit(c_s, s0) + _subsplit(e0, c_e)

    split_contents: list = []
    for c_s, c_e in contents:
        split_contents.extend(_subsplit(c_s, c_e))

    # ── 端に実静音を付与(lead/trail = min(隣接静音, edge_max))。不足分は人工静音 ──
    def _sil_before(t):     # end == t の静音 run の長さ
        for is_sil, s, e in runs:
            if is_sil and abs(e - t) < 1e-6:
                return e - s
        return 0.0
    def _sil_after(t):      # start == t の静音 run の長さ
        for is_sil, s, e in runs:
            if is_sil and abs(s - t) < 1e-6:
                return e - s
        return 0.0

    spans: List[Tuple[int, int, int, int]] = []
    for c_s, c_e in split_contents:
        prev_sil = _sil_before(c_s)
        next_sil = _sil_after(c_e)
        lead  = min(prev_sil, edge_max_sec)
        trail = min(next_sil, edge_max_sec)
        # 実静音が lead_min 未満なら人工静音で補う(先頭・末尾とも。フレーズは必ず pau で始まり
        # pau で終わる)
        art_lead  = max(0.0, lead_min_sec - prev_sil)
        art_trail = max(0.0, lead_min_sec - next_sil)
        span_s = max(0.0, c_s - lead)
        span_e = min(total_sec, c_e + trail)
        # フレームスナップ: 端が静音内にある側は静音の内側へ丸める(隣接フレーズの有音の
        # 欠片を端フレームに掴まないため)。静音が無い側は content を覆う方向。
        if lead > 0:
            s_fr = -(-int(span_s * sr) // hop_size)        # 切り上げ(静音内側)
        else:
            s_fr = int(span_s * sr) // hop_size            # 切り捨て(content を覆う)
        if trail > 0:
            e_fr = int(span_e * sr) // hop_size            # 切り捨て(静音内側)
        else:
            e_fr = -(-int(span_e * sr) // hop_size)        # 切り上げ(content を覆う)
        e_fr = min(e_fr, total_frames)
        n_art_lead  = int(round(art_lead * frame_rate))
        n_art_trail = int(round(art_trail * frame_rate))
        if e_fr > s_fr:
            spans.append((s_fr, e_fr, n_art_lead, n_art_trail))
    return spans


# ── フレーズ npz 生成 ─────────────────────────────────────────────────────────

def cut_phrases(
    name:      str,
    lab_segs:  list,
    ph_ids:    np.ndarray,   # int16  [L_ph]
    dur_sec:   np.ndarray,   # float32[L_ph]
    out_dir:   Path,
    *,
    span_data,                       # callable(s_fr, e_fr) -> (f0[n], uv[n], loud_db[n])
                                     # フレーズ単位に wav をカット→F0/ラウドネス抽出する側が渡す。
    total_frames: int,               # 曲全体フレーム数
    silence_phones = None,
    min_silence_sec:   float = _MIN_SILENCE_SEC,
    split_silence_sec: float = _SPLIT_SILENCE_SEC,
    edge_max_sec:      float = _EDGE_MAX_SEC,
    lead_min_sec:      float = _LEAD_MIN_SEC,
    max_dur_sec:       float = _MAX_DUR_SEC,
    min_loudness_db:   float = _MIN_LOUDNESS_DB,
    cl_pau_sec:        float = _CL_PAU_SEC,     # これより長い cl は静音扱い(cut)
    hop_size:          int   = 256,
    sr:                int   = 44_100,
    pau_id:            int   = 0,     # 人工静音に使う pau の vocab id
    spans:             list | None = None,   # 事前計算済み spans
    verbose:           bool  = False,
) -> int:
    """フレーズ単位で npz を書き出す。Returns: 書き出したフレーズ数。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    T_frames   = int(total_frames)
    frame_rate = sr / hop_size
    total_sec  = T_frames / frame_rate

    if spans is None:
        spans = find_phrase_spans(
            lab_segs, total_sec,
            silence_phones=silence_phones, min_silence_sec=min_silence_sec,
            split_silence_sec=split_silence_sec, edge_max_sec=edge_max_sec,
            lead_min_sec=lead_min_sec, max_dur_sec=max_dur_sec, cl_pau_sec=cl_pau_sec,
            hop_size=hop_size, sr=sr,
        )
    if not spans:
        spans = [(0, T_frames, 0, 0)]

    # 音素の cumulative フレーム境界
    cum_sec      = np.concatenate([[0.0], np.cumsum(dur_sec)])
    ph_frame_bnd = np.round(cum_sec * frame_rate).astype(int)
    ph_frame_bnd[-1] = T_frames

    n_written = 0
    n_silent  = 0
    idx = 0
    for s_fr, e_fr, n_art, n_art_trail in spans:
        if e_fr <= s_fr:
            continue
        n_frames = e_fr - s_fr

        f0_phrase_raw, uv_slice, loud_db = span_data(s_fr, e_fr)

        # ── 音量ゲート: 無音句(ラベル↔音声ズレ)を除去 ──
        if float(np.percentile(loud_db, 90)) < min_loudness_db:
            n_silent += 1
            continue

        # フレーズ単位 f0 補間(両端pauは端の voiced 値で一定、内部短pauは線形)
        voiced_phrase = uv_slice > 0.5
        if voiced_phrase.any():
            x = np.arange(n_frames, dtype=float)
            f0_slice = np.interp(x, x[voiced_phrase], f0_phrase_raw[voiced_phrase]).astype(np.float32)
        else:
            f0_slice = f0_phrase_raw.copy()

        # ── このフレーズに重なる音素を抽出 ──
        ph_mask = (ph_frame_bnd[1:] > s_fr) & (ph_frame_bnd[:-1] < e_fr)
        if not ph_mask.any():
            continue
        sel_ids   = ph_ids[ph_mask]
        sel_starts = np.maximum(ph_frame_bnd[:-1][ph_mask], s_fr)
        sel_ends   = np.minimum(ph_frame_bnd[1:][ph_mask],  e_fr)
        sel_dur_fr = np.maximum(sel_ends - sel_starts, 1)
        sel_dur_s  = (sel_dur_fr / frame_rate).astype(np.float32)
        # 末尾補正(frame スナップ由来の微差を吸収)
        diff = n_frames / frame_rate - float(sel_dur_s.sum())
        sel_dur_s[-1] = max(1.0 / frame_rate, sel_dur_s[-1] + diff)

        # ── 人工静音を端に付与(実静音 < lead_min の場合。フレーズは必ず pau で始まり終わる) ──
        # F0 は端の値でフラット延長(ゼロを混ぜない: 無声であることは uv=0 が持つ。
        # pau_aug の端延長と同じ扱いで、残差教師に -baseline の外れ値を作らない)。
        if n_art > 0:
            f0_slice  = np.concatenate([np.full(n_art, f0_slice[0], dtype=np.float32), f0_slice])
            uv_slice  = np.concatenate([np.zeros(n_art, dtype=np.float32), uv_slice])
            sel_ids   = np.concatenate([[pau_id], sel_ids])
            sel_dur_s = np.concatenate([[np.float32(n_art / frame_rate)], sel_dur_s])
        if n_art_trail > 0:
            f0_slice  = np.concatenate([f0_slice, np.full(n_art_trail, f0_slice[-1], dtype=np.float32)])
            uv_slice  = np.concatenate([uv_slice, np.zeros(n_art_trail, dtype=np.float32)])
            sel_ids   = np.concatenate([sel_ids, [pau_id]])
            sel_dur_s = np.concatenate([sel_dur_s, [np.float32(n_art_trail / frame_rate)]])

        np.savez(
            str(out_dir / f'{name}_{idx:04d}.npz'),
            phoneme_ids = sel_ids.astype(np.int16),
            dur_sec     = sel_dur_s.astype(np.float32),
            f0_interp   = f0_slice.astype(np.float32),
            uv          = uv_slice.astype(np.float32),
            # NOTE: speaker/style identity and every derived pitch quantity (nucleus values,
            # baseline) are intentionally NOT written here. The shard is identity-free and
            # opinion-free; the dataset loader derives them, so mappings and pitch parameters
            # can change without re-preprocessing.
        )
        n_written += 1
        idx += 1

    if verbose and n_silent:
        print(f'    [{name}] 無音ゲート除去 {n_silent} 句 / 書出 {n_written} 句')
    return n_written
