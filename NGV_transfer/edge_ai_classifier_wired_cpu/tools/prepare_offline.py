#!/usr/bin/env python3
"""Prepare local course assets from the seven instructor-provided raw files.

Run from the project folder:
  python tools/prepare_offline.py --list
  python tools/prepare_offline.py --download
  python tools/prepare_offline.py --from-raw
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extra_common.offline import (DEFAULT_OFFLINE_DIR, REQUIRED_FILES, SOURCES,
                                  check_source, digest, read_idx, validate_offline)


def download(source: dict, raw: Path, timeout: float, retries: int) -> None:
    target = raw / source["name"]
    if target.is_file() and digest(target, "md5") == source["md5"]:
        print(f"[reuse] {target.name}", flush=True)
        return
    for attempt in range(1, retries + 1):
        tmp = None
        try:
            print(f"[download {attempt}/{retries}] {source['url']}", flush=True)
            with tempfile.NamedTemporaryFile(dir=raw, suffix=".part", delete=False) as f:
                tmp = Path(f.name)
                req = urllib.request.Request(source["url"], headers={"User-Agent": "NGV-Offline-Preparation/1"})
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    shutil.copyfileobj(response, f, length=1024 * 1024)
            check_source(tmp, source["md5"])
            os.replace(tmp, target)
            return
        except (OSError, urllib.error.URLError, ValueError) as exc:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"다운로드 실패: {source['name']}\n{exc}\n브라우저에서 직접 받아 {raw}에 넣고 --from-raw 를 실행할 수 있습니다.") from exc
            print(f"[retry] {exc}", flush=True)


def copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as f:
            tmp = Path(f.name)
            with source.open("rb") as src:
                shutil.copyfileobj(src, f)
        os.replace(tmp, target)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def prepare(base: Path) -> None:
    import numpy as np
    raw = base / "raw"
    # Check every original before replacing any prepared outputs.
    for source in SOURCES:
        check_source(raw / source["name"], source["md5"])
    arrays = {
        "x_train": read_idx(raw / "train-images-idx3-ubyte.gz", 60000, True),
        "y_train": read_idx(raw / "train-labels-idx1-ubyte.gz", 60000, False),
        "x_test": read_idx(raw / "t10k-images-idx3-ubyte.gz", 10000, True),
        "y_test": read_idx(raw / "t10k-labels-idx1-ubyte.gz", 10000, False),
    }
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".part", delete=False) as f:
            tmp = Path(f.name)
            np.savez_compressed(f, **arrays)
        os.replace(tmp, data_dir / "fashion-mnist.npz")
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    for source in SOURCES:
        if "target" in source:
            copy_atomic(raw / source["name"], base / source["target"])
    manifest = {
        "schema": 1,
        "description": "NGV CPU-only course assets. Prepared from checksum-verified official sources.",
        "sources": list(SOURCES),
        "files": {relative: {"bytes": (base / relative).stat().st_size,
                              "sha256": digest(base / relative)} for relative in REQUIRED_FILES},
        "fashion_mnist": {name: {"shape": list(a.shape), "dtype": str(a.dtype)} for name, a in arrays.items()},
    }
    manifest_path = base / "manifest.json"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=base, suffix=".part", delete=False) as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
        manifest_tmp = Path(f.name)
    try:
        os.replace(manifest_tmp, manifest_path)
    finally:
        manifest_tmp.unlink(missing_ok=True)
    checked = validate_offline(base)
    print(f"[OK] {len(checked)}개 실습 파일 검증 완료: {base}")
    print("이 노트북의 데이터·사전학습 가중치 준비가 끝났습니다. 학습 중에는 이 로컬 파일을 사용합니다.")
    print("R5: 강사는 원본 7개만 offline/raw에 넣어 코드와 함께 ZIP으로 배포합니다. Python/패키지는 각자 온라인 설치합니다.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true", help="강사: 인터넷에서 원본 다운로드 후 준비")
    mode.add_argument("--from-raw", action="store_true", help="offline/raw 원본을 검증·변환 (인터넷 사용 없음)")
    mode.add_argument("--list", action="store_true", help="직접 다운로드할 URL·저장 파일명 표시")
    p.add_argument("--offline-dir", type=Path, default=DEFAULT_OFFLINE_DIR)
    p.add_argument("--timeout", type=float, default=60, help="각 HTTP 소켓 제한시간 (초)")
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args()
    if args.timeout <= 0 or args.retries < 1:
        p.error("--timeout > 0, --retries >= 1 이어야 합니다.")
    base = args.offline_dir.expanduser().resolve()
    if args.list:
        for source in SOURCES:
            print(f"URL: {source['url']}\n저장: {base / 'raw' / source['name']}\nMD5: {source['md5']}\n")
        return
    try:
        import numpy  # noqa: F401
        if args.download:
            raw = base / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            for source in SOURCES:
                download(source, raw, args.timeout, args.retries)
        prepare(base)
    except ImportError as exc:
        raise SystemExit("NumPy가 필요합니다. 먼저 source .venv/bin/activate 로 실습 환경을 활성화하세요.") from exc
    except Exception as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc


if __name__ == "__main__":
    main()
