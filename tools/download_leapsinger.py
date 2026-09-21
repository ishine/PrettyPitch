"""Fetch the LeapSinger acoustic models into ``checkpoints/leapsinger/``.

    python tools/download_leapsinger.py                 # latest release
    python tools/download_leapsinger.py --tag v0.1.0    # a specific one
    python tools/download_leapsinger.py --zip <path>    # a zip already downloaded

PrettyPitch only generates F0. To sing — ``svs.render``, ``notebooks/01_svs_demo.ipynb`` —
it also needs the acoustic model, which the LeapSinger project distributes as a GitHub
release. The F0 model, the consonant-length model and the NHVSing vocoder are already in
``checkpoints/``; this is the one missing piece, and it is about 100 MB.

The asset is looked up through the GitHub API rather than hard-coded, so a new release is
picked up without editing this file. Nothing is sent anywhere: the only requests are a
public API read and the download itself.

The inference **code** is a separate matter. It is imported at run time from a LeapSinger
checkout next to this repository (``../LeapSinger``), or from the path given to
``--leapsinger_root``. This script does not clone it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "checkpoints" / "leapsinger"
API = "https://api.github.com/repos/wavtechyukky/LeapSinger/releases"
NEEDED = "3speaker_gan2d.pth"          # svs.render がこの名前をこのフォルダ以下から探す


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "PrettyPitch-setup"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _find_asset(tag: str | None) -> tuple[str, str, int]:
    """(tag, download url, size) を返す。tag=None なら最新。"""
    url = API + (f"/tags/{tag}" if tag else "/latest")
    try:
        rel = json.loads(_get(url))
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"リリース情報を取得できません ({e.code} {e.reason}): {url}\n"
            "  ブラウザで https://github.com/wavtechyukky/LeapSinger/releases から zip を\n"
            "  落として、--zip <path> で渡してください。") from e
    zips = [a for a in rel.get("assets", []) if a["name"].lower().endswith(".zip")]
    if not zips:
        raise SystemExit(f"リリース {rel.get('tag_name')} に zip がありません")
    a = zips[0]
    return rel["tag_name"], a["browser_download_url"], int(a["size"])


def _download(url: str, size: int, dst: Path) -> None:
    print(f"  取得中 ({size / 1048576:.1f} MB) …", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "PrettyPitch-setup"})
    got = 0
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            if size:
                print(f"\r    {got / 1048576:6.1f} / {size / 1048576:.1f} MB"
                      f"  ({got * 100 // size:3d}%)", end="", flush=True)
    print()
    if size and got != size:
        raise SystemExit(f"サイズが合いません: {got} != {size}。もう一度実行してください")


def _extract(zip_path: Path) -> None:
    """展開してから中身を確かめ、揃っていたら DEST へ移す。途中で失敗しても既存を壊さない。"""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "x"
        with zipfile.ZipFile(zip_path) as z:
            bad = [n for n in z.namelist() if n.startswith("/") or ".." in Path(n).parts]
            if bad:
                raise SystemExit(f"zip に不正なパスが入っています: {bad[:3]}")
            z.extractall(work)
        hits = sorted(work.rglob(NEEDED))
        if not hits:
            raise SystemExit(f"zip の中に {NEEDED} が見つかりません")
        root = hits[0].parent
        DEST.mkdir(parents=True, exist_ok=True)
        out = DEST / root.name
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(root), str(out))
        print(f"  展開しました -> {out.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None, help="リリースのタグ（既定: 最新）")
    ap.add_argument("--zip", default=None, help="ダウンロード済みの zip を使う")
    ap.add_argument("--force", action="store_true", help="既に置いてあっても取り直す")
    a = ap.parse_args()

    have = sorted(DEST.rglob(NEEDED)) if DEST.is_dir() else []
    if have and not a.force:
        print(f"すでにあります: {have[0].relative_to(REPO)}")
        print("  取り直すなら --force")
        return

    if a.zip:
        _extract(Path(a.zip))
    else:
        tag, url, size = _find_asset(a.tag)
        print(f"LeapSinger {tag}")
        with tempfile.TemporaryDirectory() as tmp:
            zp = Path(tmp) / "models.zip"
            _download(url, size, zp)
            _extract(zp)

    hits = sorted(DEST.rglob(NEEDED))
    if not hits:
        raise SystemExit(f"{NEEDED} を配置できませんでした")
    print(f"\n完了: {hits[0].relative_to(REPO)}")
    print("推論コードは別途、このリポジトリの隣に LeapSinger を clone してください:")
    print("  git clone https://github.com/wavtechyukky/LeapSinger.git ../LeapSinger")


if __name__ == "__main__":
    sys.exit(main())
