import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.check_capture import inspect_capture


class CaptureCheckTests(unittest.TestCase):
    def test_complete_then_missing(self):
        cfg = {'labels': ['cup', 'pen'], 'sessions': ['s01', 's02'], 'photos_per_session': 2}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for label in cfg['labels']:
                for session in cfg['sessions']:
                    folder = root/label/session
                    folder.mkdir(parents=True)
                    for i in range(2): (folder/f'{i}.jpg').write_bytes(b'test')
            _, errors, total = inspect_capture(root, cfg)
            self.assertEqual(errors, [])
            self.assertEqual(total, 8)
            (root/'pen/s02/1.jpg').unlink()
            self.assertEqual(inspect_capture(root, cfg)[1], ['pen/s02: 1 / 2'])

    def test_old_label_and_extra_format_rejected(self):
        cfg = {'labels': ['cup'], 'sessions': ['s01'], 'photos_per_session': 1}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'cup/s01').mkdir(parents=True)
            (root/'cup/s01/0.jpg').write_bytes(b'test')
            (root/'old_label').mkdir()
            (root/'old_label/extra.jpg').write_bytes(b'test')
            (root/'cup/s01/extra.JPG').write_bytes(b'test')
            self.assertEqual(len(inspect_capture(root, cfg)[1]), 2)


if __name__ == '__main__': unittest.main()
