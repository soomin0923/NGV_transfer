"""Summarize fresh camera samples without assuming custom labels equal ImageNet.

Provide an explicit JSON mapping for semantic ImageNet scores, for example
{"cup": ["cup", "coffee_mug"], "bottle": ["water_bottle", "pop_bottle"]}.
Names on the right must exactly match the pretrained model's labels.txt. Omitted
classes are not scored against ImageNet. Out-of-scope samples are never scored.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from collections import defaultdict
from statistics import median
import math
from extra_common.util import ROOT, save_csv, write_json, labels
from extra_common.course import load_config

# An example only. It is never applied implicitly to a student's chosen labels.
MAPPING = {'cup':{'cup', 'coffee_mug'}, 'bottle':{'water_bottle', 'pop_bottle', 'wine_bottle', 'beer_bottle'},
           'wallet':{'wallet'}, 'remote':{'remote_control'}}


def summarize(rows, mapping=None, class_names=None):
    mapping = {} if mapping is None else mapping
    allowed = set(class_names) if class_names is not None else {r['truth'] for r in rows} - {'out_of_scope'}
    groups = defaultdict(list)
    for row in rows:
        if row['truth'] in allowed and row['truth'] != 'out_of_scope':
            groups[(row['truth'], row['condition'])].append(row)
    out = []
    for (truth, condition), items in sorted(groups.items()):
        paired = [r for r in items if r.get('custom_top1') and r.get('imagenet_top1')]
        custom = [r for r in items if r.get('custom_top1')]
        imagenet = [r for r in items if r.get('imagenet_top1')]
        aliases = set(mapping.get(truth, []))
        out.append({'truth':truth, 'condition':condition, 'samples':len(items), 'paired_samples':len(paired),
            'custom_samples':len(custom), 'imagenet_samples':len(imagenet), 'imagenet_mapping_defined':bool(aliases),
            'custom_accuracy':sum(r['custom_top1'] == truth for r in custom) / len(custom) if custom else None,
            'custom_accuracy_paired':sum(r['custom_top1'] == truth for r in paired) / len(paired) if paired else None,
            'imagenet_top1_mapped_accuracy':sum(r['imagenet_top1'] in aliases for r in imagenet) / len(imagenet) if aliases and imagenet else None,
            'imagenet_top1_mapped_accuracy_paired':sum(r['imagenet_top1'] in aliases for r in paired) / len(paired) if aliases and paired else None,
            'imagenet_top5_contains_mapped_class':sum(bool(set(r.get('imagenet_top5', '').split('|')) & aliases) for r in imagenet) / len(imagenet) if aliases and imagenet else None})
    return out



MODEL_FIELDS = ('custom_model_sha256', 'imagenet_model_sha256')
GROUP_FIELDS = MODEL_FIELDS + ('hostname', 'machine', 'threads', 'custom_execution_path', 'imagenet_execution_path')

def summarize_groups(rows, mapping=None, class_names=None):
    """Keep different known model files separate; legacy unknowns remain marked."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key) or 'unrecorded') for key in GROUP_FIELDS)].append(row)
    out = []
    for number, (key, items) in enumerate(sorted(groups.items()), start=1):
        for summary in summarize(items, mapping, class_names):
            summary.update(model_group=number, **dict(zip(GROUP_FIELDS, key)))
            summary['custom_model'] = items[0].get('custom_model') or 'unrecorded'
            summary['imagenet_model'] = items[0].get('imagenet_model') or 'unrecorded'
            selected = [r for r in items if r['truth'] == summary['truth'] and r['condition'] == summary['condition']]
            for model in ['custom', 'imagenet']:
                values = []
                for row in selected:
                    value = row.get(model + '_inference_ms')
                    if value not in (None, ''):
                        v = float(value)
                        if math.isfinite(v) and v >= 0: values.append(v)
                summary[model + '_observed_ms_samples'] = len(values)
                summary[model + '_observed_median_ms'] = float(median(values)) if values else None
            out.append(summary)
    return out

def load_mapping(path, class_names, imagenet_label_file):
    doc = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(doc, dict):
        raise ValueError('Mapping JSON must be an object: custom label -> nonempty list of exact ImageNet labels')
    known = set(labels(imagenet_label_file, 1000, allow_duplicates=True))
    for key, value in doc.items():
        if key not in class_names:
            raise ValueError(f'Mapping contains a label absent from course_config.json: {key}')
        if not isinstance(value, list) or not value or any(not isinstance(s, str) for s in value):
            raise ValueError(f'{key}: use a nonempty list of exact ImageNet label strings')
        unknown = set(value) - known
        if unknown:
            raise ValueError(f'{key}: names absent from ImageNet labels.txt: {sorted(unknown)}')
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default=str(ROOT / 'day2_extra/comparison/samples.csv'))
    p.add_argument('--config', default=None)
    p.add_argument('--mapping', help='Explicit custom label -> ImageNet labels JSON; omitted classes have no semantic score')
    p.add_argument('--imagenet-labels', default=str(ROOT / 'day2_extra/artifacts/labels.txt'))
    a = p.parse_args()
    class_names = load_config(a.config)['labels']
    mapping = load_mapping(a.mapping, class_names, a.imagenet_labels) if a.mapping else {}
    with open(a.input, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    out = summarize_groups(rows, mapping, class_names)
    if not out:
        raise SystemExit('No configured-class comparison samples. Record them in the browser first.')
    save_csv(Path(a.input).parent / 'summary.csv', out)
    write_json(Path(a.input).parent / 'summary.json', {'groups':out, 'explicit_imagenet_mapping':mapping,
        'out_of_scope_or_unknown_not_scored':sum(r['truth'] not in class_names for r in rows),
        'note':'ImageNet semantic scores require an explicit correspondence. Top-5 coverage differs from custom Top-1 accuracy. Compare paired Top-1 scores on the same fresh samples.'})
    if not mapping:
        print('No --mapping supplied: ImageNet semantic accuracy is intentionally blank; custom exact-label scores are available.')
    print(out)


if __name__ == '__main__':
    main()
