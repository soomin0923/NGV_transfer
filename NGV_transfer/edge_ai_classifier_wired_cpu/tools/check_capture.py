"""Check every configured camera label/session before copying or training."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extra_common.course import ROOT, load_config


def inspect_capture(root, cfg):
    root = Path(root)
    errors = []
    rows = []
    expected_paths = set()
    for label in cfg['labels']:
        counts = []
        for session in cfg['sessions']:
            folder = root / label / session
            photos = list(folder.glob('*.jpg'))
            expected_paths.update(photos)
            counts.append(len(photos))
            if len(photos) != cfg['photos_per_session']:
                errors.append(f"{label}/{session}: {len(photos)} / {cfg['photos_per_session']}")
        rows.append((label, counts))
    all_photos = {p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
    unexpected = sorted(all_photos - expected_paths)
    for p in unexpected[:20]:
        errors.append('Unexpected image: ' + str(p.relative_to(root)))
    if len(unexpected) > 20:
        errors.append(f'... and {len(unexpected)-20} more unexpected images')
    return rows, errors, len(expected_paths)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'dataset_raw')
    p.add_argument('--config', type=Path, default=ROOT/'course_config.json')
    a = p.parse_args()
    cfg = load_config(a.config)
    rows, errors, total = inspect_capture(a.data, cfg)
    print('Sessions:', ', '.join(cfg['sessions']))
    for label, counts in rows:
        print(label, counts)
    if errors:
        raise SystemExit('FAIL: 촬영 누락·초과 또는 다른 라벨의 파일을 확인하세요.\n' + '\n'.join(errors))
    print(f'PASS: {total} photos; every label/session has {cfg["photos_per_session"]}.')


if __name__ == '__main__':
    main()
