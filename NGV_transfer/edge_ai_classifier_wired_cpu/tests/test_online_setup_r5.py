"""Installer regressions: missing classroom assets must stop before internet/apt."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RAW = (
    'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz',
    'weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5',
    'weights_mobilenet_v3_small_224_1.0_float.h5', 'imagenet_class_index.json',
)


class LaptopPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        (self.project / 'tools').mkdir()
        (self.project / 'offline/raw').mkdir(parents=True)
        shutil.copy2(ROOT / 'setup_laptop.sh', self.project / 'setup_laptop.sh')
        # Controlled OS/user stubs: only the preflight orchestration is under test.
        (self.project / 'tools/wsl_common.sh').write_text(
            'ngv_user() { :; }\nngv_fail() { echo "STOP: $*" >&2; exit 1; }\n')
        (self.project / 'tools/check_wsl_host.sh').write_text('echo "HOST CHECK PASS (test stub)"\n')
        (self.project / 'tools/install_wsl_os_deps.sh').write_text('touch apt_was_reached\nexit 23\n')

    def run_setup(self):
        return subprocess.run(['bash', 'setup_laptop.sh'], cwd=self.project,
                              capture_output=True, text=True)

    def test_missing_data_stops_before_apt_and_lists_every_filename(self):
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.project / 'apt_was_reached').exists())
        for name in RAW:
            self.assertIn('Missing: offline/raw/' + name, result.stderr)

    def test_one_missing_or_empty_file_also_stops(self):
        for name in RAW:
            (self.project / 'offline/raw' / name).write_bytes(b'test')
        (self.project / 'offline/raw' / RAW[-1]).write_bytes(b'')
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.project / 'apt_was_reached').exists())
        self.assertIn(RAW[-1], result.stderr)

    def test_all_seven_files_allow_os_setup_phase(self):
        for name in RAW:
            (self.project / 'offline/raw' / name).write_bytes(b'test')
        result = self.run_setup()
        self.assertEqual(result.returncode, 23)
        self.assertTrue((self.project / 'apt_was_reached').is_file())

    @unittest.skipUnless(hasattr(os, 'geteuid') and os.geteuid() == 0, 'root-only guard check')
    def test_real_entrypoints_reject_root_before_any_install(self):
        for script in ['setup_laptop.sh', 'setup_pi.sh']:
            result = subprocess.run(['bash', str(ROOT / script)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('ordinary', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
