"""Evaluate the three exported camera models on validation and test photos."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def evaluation_commands(data, artifacts):
    variants = (
        ('float32_frozen', artifacts/'model_float32_frozen.tflite', artifacts/'labels.txt'),
        ('dynamic_frozen', artifacts/'dynamic_diagnostic/model_dynamic_frozen.tflite', artifacts/'dynamic_diagnostic/labels.txt'),
        ('int8_frozen', artifacts/'int8/model_int8_frozen.tflite', artifacts/'int8/labels.txt'),
    )
    missing = [str(p) for _, model, labels in variants for p in (model, labels) if not p.is_file()]
    missing += [str(data/split) for split in ('val', 'test') if not (data/split).is_dir()]
    if missing:
        raise ValueError('먼저 데이터 분할과 FLOAT32/DYNAMIC/INT8 변환을 완료하세요.\nMissing:\n' + '\n'.join(missing))
    for split, prefix in (('val', 'val'), ('test', 'eval')):
        for name, model, labels in variants:
            yield f'{split} / {name}', [sys.executable, str(ROOT/'wsl/evaluate_tflite.py'),
                '--model', str(model), '--data', str(data/split), '--labels', str(labels),
                '--output', str(artifacts/f'{prefix}_{name}')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'dataset_split')
    p.add_argument('--artifacts', type=Path, default=ROOT/'artifacts')
    a = p.parse_args()
    try:
        commands = list(evaluation_commands(a.data.expanduser().resolve(), a.artifacts.expanduser().resolve()))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for i, (label, command) in enumerate(commands, 1):
        print(f'[{i}/{len(commands)}] {label}', flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
    print('PASS: three precisions evaluated on both validation and test photos.')


if __name__ == '__main__':
    main()
