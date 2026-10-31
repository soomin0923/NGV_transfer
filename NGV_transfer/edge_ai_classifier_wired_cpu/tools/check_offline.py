#!/usr/bin/env python3
"""Validate locally copied assets; never access the network."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extra_common.offline import DEFAULT_OFFLINE_DIR, validate_offline


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-dir", type=Path, default=DEFAULT_OFFLINE_DIR)
    args = p.parse_args()
    try:
        files = validate_offline(args.offline_dir.expanduser().resolve())
    except ImportError as exc:
        raise SystemExit("NumPy가 필요합니다. source .venv/bin/activate 후 다시 실행하세요.") from exc
    except Exception as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc
    for relative in files:
        print(f"[OK] {relative}")
    print("오프라인 데이터/사전학습 가중치 검증 완료. 네트워크를 사용하지 않았습니다.")


if __name__ == "__main__":
    main()
