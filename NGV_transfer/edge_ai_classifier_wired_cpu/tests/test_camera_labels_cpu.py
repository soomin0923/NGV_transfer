"""Small device-free tests for configurable capture, leakage checks and mappings."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from extra_common.course import DEFAULTS
from extra_common.camera import Camera, camera_args, options
from extra_common.live_camera import LiveApp
from extra_common.util import write_json
from wsl.prepare_dataset import prepare_dataset
from day2_extra.summarize_comparison import summarize, load_mapping


class ConfiguredCaptureTests(unittest.TestCase):
    def test_course_and_saved_camera_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Path(folder) / 'course.json'
            saved = Path(folder) / 'camera.json'
            write_json(cfg, {'camera':{'camera':'3', 'width':320, 'height':240, 'fps':12}})
            write_json(saved, {'requested':{'camera':'5', 'width':640, 'height':480, 'fps':15, 'fourcc':'MJPG'}})
            parser = argparse.ArgumentParser()
            camera_args(parser)
            result = options(parser.parse_args(['--config', str(cfg), '--camera-config', str(saved), '--fps', '10']))
            self.assertEqual(result, {'camera':'5', 'width':640, 'height':480, 'fps':10, 'fourcc':'MJPG'})

    def test_unreadable_native_camera_uses_fallback(self):
        import cv2
        captures = []
        class Fake:
            def __init__(self, *args):
                self.index = len(captures)
                self.settings = []
                self.released = False
                captures.append(self)
            def isOpened(self): return True
            def set(self, *args): self.settings.append(args); return True
            def get(self, key): return 15 if key == cv2.CAP_PROP_FPS else 0
            def read(self):
                if self.index == 0: return False, None
                return True, np.zeros((240, 320, 3), np.uint8)
            def release(self): self.released = True
        with patch('cv2.VideoCapture', Fake), patch('extra_common.camera.time.sleep'):
            cam = Camera({'camera':'0', 'width':0, 'height':0, 'fps':0, 'fourcc':'AUTO'})
            self.assertEqual(cam.requested['fourcc'], 'MJPG')
            self.assertEqual(cam.actual['width'], 320)
            self.assertTrue(captures[0].released)
            cam.close()

    def test_custom_order_and_capture_count(self):
        class FakeCamera:
            actual = {'width':123, 'height':77, 'device':'mock'}
            def __init__(self, *args): pass
            def read(self): return np.full((77, 123, 3), 80, np.uint8)
            def close(self): pass
        cfg = copy.deepcopy(DEFAULTS)
        cfg['labels'] = ['zebra_object', 'apple_object']
        cfg['photos_per_session'] = 2
        with tempfile.TemporaryDirectory() as folder, patch('extra_common.live_camera.Camera', FakeCamera):
            app = LiveApp({}, output=folder, interval=.01, config=cfg)
            try:
                self.assertEqual(app.cls, 'zebra_object')
                self.assertIn('apple_object', app.page('token'))
                self.assertNotIn('<option>cup</option>', app.page('token'))
                app.action('/select', {'class':'apple_object', 'session':'s05'})
                app.action('/start', {})
                deadline = time.monotonic() + 3
                while app.count() < 2 and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(app.count(), 2)
                self.assertFalse(app.state()['collecting'])
                with self.assertRaises(ValueError):
                    app.action('/select', {'class':'../bad', 'session':'s01'})
            finally:
                app.close()


class DatasetTests(unittest.TestCase):
    def create_captures(self, root):
        cfg = copy.deepcopy(DEFAULTS)
        cfg['labels'] = ['zebra_object', 'apple_object']
        cfg['photos_per_session'] = 2
        counter = 0
        for cls in cfg['labels']:
            for session in cfg['sessions']:
                folder = root / cls / session
                folder.mkdir(parents=True)
                for index in range(2):
                    counter += 1
                    Image.fromarray(np.full((24, 24, 3), counter * 7, np.uint8)).save(folder / f'{index}.jpg')
        return cfg

    def test_order_counts_and_session_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            root, dst = Path(folder) / 'raw', Path(folder) / 'split'
            cfg = self.create_captures(root)
            rows = prepare_dataset(root, dst, cfg)
            self.assertEqual((dst / 'class_names.txt').read_text().splitlines(), cfg['labels'])
            self.assertEqual({s:sum(r['split'] == s for r in rows) for s in ['train', 'val', 'test']}, {'train':12, 'val':4, 'test':4})
            self.assertEqual({r['session'] for r in rows if r['split'] == 'test'}, {'s05'})

    def test_invalid_count_leaves_no_partial_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root, dst = Path(folder) / 'raw', Path(folder) / 'split'
            cfg = self.create_captures(root)
            (root / 'apple_object/s05/1.jpg').unlink()
            with self.assertRaisesRegex(ValueError, 'expected 2'):
                prepare_dataset(root, dst, cfg)
            self.assertFalse(dst.exists())

    def test_exact_duplicate_across_splits_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root, dst = Path(folder) / 'raw', Path(folder) / 'split'
            cfg = self.create_captures(root)
            (root / 'apple_object/s05/0.jpg').write_bytes((root / 'apple_object/s01/0.jpg').read_bytes())
            with self.assertRaisesRegex(ValueError, 'Identical image across'):
                prepare_dataset(root, dst, cfg)
            self.assertFalse(dst.exists())

    def test_unconfigured_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root, dst = Path(folder) / 'raw', Path(folder) / 'split'
            cfg = self.create_captures(root)
            Image.new('RGB', (24, 24)).save(root / 'stray.jpg')
            with self.assertRaisesRegex(ValueError, 'Unexpected image'):
                prepare_dataset(root, dst, cfg)
            self.assertFalse(dst.exists())


class ComparisonTests(unittest.TestCase):
    def test_custom_labels_have_no_implicit_imagenet_score(self):
        rows = [{'truth':'my_tool', 'condition':'bright', 'custom_top1':'my_tool', 'imagenet_top1':'hammer', 'imagenet_top5':'hammer|nail'},
                {'truth':'out_of_scope', 'condition':'bright'}]
        row = summarize(rows, class_names=['my_tool', 'red_cup'])[0]
        self.assertEqual(row['custom_accuracy_paired'], 1)
        self.assertIsNone(row['imagenet_top1_mapped_accuracy'])
        row = summarize(rows, {'my_tool':['hammer']}, ['my_tool', 'red_cup'])[0]
        self.assertEqual(row['imagenet_top1_mapped_accuracy_paired'], 1)

    def test_mapping_rejects_labels_absent_from_model(self):
        with tempfile.TemporaryDirectory() as folder:
            names = Path(folder) / 'labels.txt'
            names.write_text('\n'.join(f'name_{i}' for i in range(1000)))
            mapping = Path(folder) / 'mapping.json'
            write_json(mapping, {'my_tool':['imaginary_label']})
            with self.assertRaisesRegex(ValueError, 'absent from ImageNet'):
                load_mapping(mapping, ['my_tool', 'red_cup'], names)


if __name__ == '__main__':
    unittest.main()
