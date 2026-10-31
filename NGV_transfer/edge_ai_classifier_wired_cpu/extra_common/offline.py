"""Local-only course assets and integrity checks (never downloads)."""
from __future__ import annotations

import gzip
import hashlib
import json
import struct
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFLINE_DIR = PROJECT_ROOT / "offline"
FASHION_BASE = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
WEIGHTS_BASE = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/"
# Official Keras 3 source uses *_no_top_v2.h5, NOT the old TF 2 no_top.h5.
# Sources: keras/src/applications/mobilenet_v3.py and imagenet_utils.py;
# Fashion-MNIST compressed-file MD5s: zalandoresearch/fashion-mnist README.
SOURCES = (
    {"name": "train-images-idx3-ubyte.gz", "url": FASHION_BASE + "train-images-idx3-ubyte.gz", "md5": "8d4fb7e6c68d591d4c3dfef9ec88bf0d"},
    {"name": "train-labels-idx1-ubyte.gz", "url": FASHION_BASE + "train-labels-idx1-ubyte.gz", "md5": "25c81989df183df01b3e8a0aad5dffbe"},
    {"name": "t10k-images-idx3-ubyte.gz", "url": FASHION_BASE + "t10k-images-idx3-ubyte.gz", "md5": "bef4ecab320f06d8554ea6380940ec79"},
    {"name": "t10k-labels-idx1-ubyte.gz", "url": FASHION_BASE + "t10k-labels-idx1-ubyte.gz", "md5": "bb300cfdad3c16e7a12a480ee83cd310"},
    {"name": "weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5", "url": WEIGHTS_BASE + "weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5", "md5": "d3e8ec802a04aa4fc771ee12a9a9b836", "target": "weights/mobilenet_v3_small_notop.h5"},
    {"name": "weights_mobilenet_v3_small_224_1.0_float.h5", "url": WEIGHTS_BASE + "weights_mobilenet_v3_small_224_1.0_float.h5", "md5": "8768d4c2e7dee89b9d02b2d03d65d862", "target": "weights/mobilenet_v3_small_top.h5"},
    {"name": "imagenet_class_index.json", "url": "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json", "md5": "c2c37ea517e94d9795004a39431a14cb", "target": "imagenet_class_index.json"},
)
REQUIRED_FILES = ("data/fashion-mnist.npz", "weights/mobilenet_v3_small_notop.h5", "weights/mobilenet_v3_small_top.h5", "imagenet_class_index.json")
RECOVERY = "강사가 배포한 ZIP의 원본 7개를 offline/raw에 넣고 bash setup_laptop.sh를 실행하세요. 환경 설치 후 데이터만 다시 준비하려면 source day3/env_day2.sh 다음 python tools/prepare_offline.py --from-raw를 실행하세요."


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_asset(relative: str, offline_dir: Path | str | None = None) -> Path:
    """Resolve a file under offline/; no network or Keras cache fallback."""
    base = Path(offline_dir) if offline_dir is not None else DEFAULT_OFFLINE_DIR
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("offline 경로 밖의 파일은 사용할 수 없습니다.")
    if not path.is_file():
        raise FileNotFoundError(f"오프라인 파일이 없습니다: {path}\n{RECOVERY}")
    return path


def check_source(path: Path, expected_md5: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"원본 파일이 없습니다: {path}\npython tools/prepare_offline.py --list 로 다운로드 주소를 확인하세요.")
    actual = digest(path, "md5")
    if actual != expected_md5:
        raise ValueError(f"파일이 손상되었거나 다른 버전입니다: {path}\nMD5: {actual}; expected: {expected_md5}\n해당 원본을 다시 다운로드하세요.")


def read_idx(path: Path, expected_count: int, images: bool):
    import numpy as np
    header_size = 16 if images else 8
    expected_shape = (expected_count, 28, 28) if images else (expected_count,)
    expected_bytes = int(np.prod(expected_shape))
    with gzip.open(path, "rb") as f:
        header = f.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"잘린 IDX 헤더: {path}")
        numbers = struct.unpack(">IIII" if images else ">II", header)
        expected_header = (2051, expected_count, 28, 28) if images else (2049, expected_count)
        if numbers != expected_header:
            raise ValueError(f"잘못된 IDX 헤더: {path}: {numbers} (expected {expected_header})")
        payload = f.read(expected_bytes + 1)
    if len(payload) != expected_bytes:
        raise ValueError(f"잘못된 IDX 데이터 길이: {path}: {len(payload)} (expected {expected_bytes})")
    return np.frombuffer(payload, dtype=np.uint8).reshape(expected_shape).copy()


def validate_fashion(path: Path) -> dict:
    import numpy as np
    expected = {"x_train": (60000, 28, 28), "y_train": (60000,), "x_test": (10000, 28, 28), "y_test": (10000,)}
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(expected):
            raise ValueError(f"Fashion-MNIST 키가 다릅니다: {z.files}")
        for key, shape in expected.items():
            array = z[key]
            if array.shape != shape or array.dtype != np.uint8:
                raise ValueError(f"{key}: shape/dtype={array.shape}/{array.dtype}; expected {shape}/uint8")
            if key.startswith("y"):
                counts = np.bincount(array.astype(np.int64), minlength=10)
                if len(counts) != 10 or not np.all(counts == shape[0] // 10):
                    raise ValueError(f"{key}: 원본 Fashion-MNIST의 균형 잡힌 10개 클래스가 아닙니다.")
    return {key: list(shape) for key, shape in expected.items()}


def validate_class_index(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {str(i) for i in range(1000)}:
        raise ValueError("ImageNet class index must contain keys 0..999.")
    if any(not isinstance(v, list) or len(v) != 2 or not all(isinstance(s, str) and s for s in v) for v in data.values()):
        raise ValueError("ImageNet class index entries must be [synset, label].")


def validate_offline(offline_dir: Path | str | None = None) -> list[str]:
    """Verify manifest SHA-256, official source MD5, and dataset schema."""
    base = Path(offline_dir) if offline_dir is not None else DEFAULT_OFFLINE_DIR
    manifest_path = require_asset("manifest.json", base)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError(f"지원하지 않는 manifest: {manifest_path}")
    checked = []
    for relative in REQUIRED_FILES:
        path = require_asset(relative, base)
        record = manifest["files"].get(relative)
        if not isinstance(record, dict) or not record.get("sha256"):
            raise ValueError(f"manifest에 {relative}가 없습니다. {RECOVERY}")
        if path.stat().st_size != record.get("bytes") or digest(path) != record["sha256"]:
            raise ValueError(f"SHA-256/크기가 다릅니다: {path}. {RECOVERY}")
        checked.append(relative)
    for source in SOURCES:
        if "target" in source:
            path = base / source["target"]
            check_source(path, source["md5"])
            if path.suffix == ".h5":
                with path.open("rb") as f:
                    if f.read(8) != b"\x89HDF\r\n\x1a\n":
                        raise ValueError(f"HDF5 헤더가 아닙니다: {path}")
        # raw files are optional after preparation, but if retained must be valid.
        raw = base / "raw" / source["name"]
        if raw.exists():
            check_source(raw, source["md5"])
    validate_fashion(base / "data/fashion-mnist.npz")
    validate_class_index(base / "imagenet_class_index.json")
    return checked
