"""Device-free integration of real recording, CSV upgrade, and analysis grouping."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import csv
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from extra_common.course import DEFAULTS
from extra_common.live_camera import LiveApp
from extra_common.util import sha256
from day2_extra.analyze_comparison import grouped_summary


class RecordingUpgradeTests(unittest.TestCase):
    def test_r1_upgrade_preserves_records_and_separates_unknown_provenance(self):
        """Real CSV/JPEG writes; only camera, inference and thread start are mocked."""
        class FakeCamera:
            actual = {'device': 'test-only-no-hardware', 'width': 80, 'height': 60}
            def __init__(self, *_): pass
            def close(self): pass
        def fake_runner(*_):
            return SimpleNamespace(execution_path='native_batch')
        config = copy.deepcopy(DEFAULTS)
        config['labels'] = ['cup', 'other']
        config['threads'] = 2
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            comparison = root / 'day2_extra/comparison'
            comparison.mkdir(parents=True)
            csv_path = comparison / 'samples.csv'
            old_row = {'image': 'r1_original.jpg', 'truth': 'cup', 'condition': 'normal',
                       'frame_sequence': '9', 'custom_top1': 'other',
                       'custom_confidence': '0.7', 'custom_top5': 'other|cup',
                       'imagenet_top1': 'coffee_mug', 'imagenet_confidence': '0.8',
                       'imagenet_top5': 'coffee_mug|cup'}
            with csv_path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(old_row))
                writer.writeheader(); writer.writerow(old_row)
            original_csv = csv_path.read_bytes()
            custom = root / 'custom_fp32.tflite'; custom.write_bytes(b'not-a-model: mocked inference custom')
            imagenet = root / 'imagenet_int8.tflite'; imagenet.write_bytes(b'not-a-model: mocked inference imagenet')
            custom_labels = root / 'custom_labels.txt'; custom_labels.write_text('cup\nother\n')
            imagenet_labels = root / 'imagenet_labels.txt'
            imagenet_labels.write_text('\n'.join(['coffee_mug', 'cup'] + [f'fixture_{i}' for i in range(998)]) + '\n')
            environment = {'hostname': 'fixture-pi', 'machine': 'aarch64'}
            with patch('extra_common.live_camera.ROOT', root), \
                 patch('extra_common.live_camera.Camera', FakeCamera), \
                 patch('extra_common.live_camera.LiteRunner', fake_runner), \
                 patch('extra_common.live_camera.threading.Thread'), \
                 patch('extra_common.live_camera.metadata', return_value=environment):
                app = LiveApp({}, mode='imagenet', model=custom, label_file=custom_labels,
                              imagenet_model=imagenet, imagenet_labels=imagenet_labels, config=config)
                try:
                    app.raw = np.full((60, 80, 3), 127, dtype=np.uint8)
                    app.sequence = 10
                    app.last = {
                        'custom': {'top1': 'cup', 'confidence': .9, 'inference_ms': 2.5,
                                   'top5': [{'label': 'cup', 'probability': .9}, {'label': 'other', 'probability': .1}]},
                        'imagenet': {'top1': 'coffee_mug', 'confidence': .8, 'inference_ms': 3.5,
                                     'top5': [{'label': 'coffee_mug', 'probability': .8}]}}
                    self.assertEqual(app.action('/record', {'truth': 'cup', 'condition': 'normal'}), {'ok': True})
                    with csv_path.open(newline='', encoding='utf-8') as stream:
                        reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
                    self.assertEqual(len(rows), 2)
                    for key, value in old_row.items(): self.assertEqual(rows[0][key], value)
                    self.assertEqual(rows[0]['custom_model_sha256'], '')
                    self.assertEqual(rows[0]['imagenet_model_sha256'], '')
                    self.assertEqual(rows[0]['hostname'], '')
                    latest = rows[1]
                    self.assertEqual(latest['custom_model_sha256'], sha256(custom))
                    self.assertEqual(latest['imagenet_model_sha256'], sha256(imagenet))
                    self.assertEqual(latest['custom_model'], custom.name)
                    self.assertEqual(latest['imagenet_model'], imagenet.name)
                    self.assertEqual(latest['hostname'], 'fixture-pi')
                    self.assertEqual(latest['machine'], 'aarch64')
                    self.assertEqual(latest['threads'], '2')
                    self.assertEqual(latest['custom_execution_path'], 'native_batch')
                    self.assertEqual(latest['imagenet_execution_path'], 'native_batch')
                    self.assertEqual(float(latest['custom_inference_ms']), 2.5)
                    self.assertEqual(float(latest['imagenet_inference_ms']), 3.5)
                    self.assertTrue((comparison / 'images' / latest['image']).is_file())
                    self.assertTrue(all(None not in row and all(value is not None for value in row.values()) for row in rows))
                    backup = list(comparison.glob('samples_before_r2_*.csv'))
                    self.assertEqual(len(backup), 1)
                    self.assertEqual(backup[0].read_bytes(), original_csv)
                    # A duplicate frame cannot append another CSV row or image.
                    with self.assertRaisesRegex(ValueError, 'already recorded'):
                        app.action('/record', {'truth': 'cup', 'condition': 'normal'})
                    self.assertEqual(len(list((comparison / 'images').glob('*.jpg'))), 1)
                    # Next append uses the expanded header without another upgrade;
                    # a changed execution path stays separate in the analyzer.
                    app.sequence = 11
                    app.runners['custom'].execution_path = 'per_sample_fallback'
                    app.action('/record', {'truth': 'cup', 'condition': 'normal'})
                    with csv_path.open(newline='', encoding='utf-8') as stream:
                        reader = csv.DictReader(stream)
                        self.assertEqual(reader.fieldnames, fields)
                        rows = list(reader)
                    self.assertEqual(len(rows), 3)
                    self.assertEqual(len(list(comparison.glob('samples_before_r2_*.csv'))), 1)
                    groups = grouped_summary(rows, {'cup': ['cup', 'coffee_mug']}, ['cup', 'other'])
                    self.assertEqual(len(groups), 3)
                    self.assertEqual(sorted(row['samples'] for row in groups), [1, 1, 1])
                    legacy = next(row for row in groups if row['custom_model_sha256'] == 'unrecorded')
                    self.assertEqual(legacy['custom_accuracy_paired'], 0)
                    self.assertIsNone(legacy['custom_observed_median_ms'])
                    modern = [row for row in groups if row['custom_model_sha256'] == sha256(custom)]
                    self.assertEqual(len(modern), 2)
                    self.assertEqual({row['custom_execution_path'] for row in modern}, {'native_batch', 'per_sample_fallback'})
                    self.assertTrue(all(row['custom_accuracy_paired'] == 1 and row['imagenet_top1_mapped_accuracy_paired'] == 1 for row in modern))
                finally:
                    app.close()


if __name__ == '__main__': unittest.main()
