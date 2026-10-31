"""Meaningful bootstrap regressions: ABI rejection, tamper detection, lock consistency."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wsl_bundle_tested', ROOT / 'tools/wsl_bundle.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make_wheel(path, name='demo', version='1.0', elf=None):
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(f'{name}-{version}.dist-info/METADATA', f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
        if elf is not None:
            z.writestr('demo/native.so', b'\x7fELF' + elf)


class WheelCompatibilityTests(unittest.TestCase):
    def test_wheels_accept_only_target_abi(self):
        good = ['demo-1.0-cp312-cp312-manylinux_2_27_x86_64.whl',
                'demo-1.0-cp310-abi3-manylinux2014_x86_64.whl', 'demo-1.0-py3-none-any.whl']
        bad = ['demo-1.0-cp312-cp312-manylinux_2_28_x86_64.whl',
               'demo-1.0-cp312-cp312-linux_x86_64.whl', 'demo-1.0-cp312-cp312-manylinux_2_27_aarch64.whl',
               'demo-1.0-cp311-cp311-manylinux_2_17_x86_64.whl', 'demo-1.0-cp313-abi3-manylinux_2_17_x86_64.whl']
        for name in good:
            self.assertTrue(mod.compatible_wheel(name), name)
        for name in bad:
            self.assertFalse(mod.compatible_wheel(name), name)

    def test_elf_rejects_under_tagged_newer_libc(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / 'demo-1.0-py3-none-any.whl'
            make_wheel(wheel, elf=b'GLIBC_2.28\x00GLIBCXX_3.4.22\x00')
            with self.assertRaisesRegex(ValueError, 'above the 2.27'):
                mod.audit_elf(wheel)
            make_wheel(wheel, elf=b'GLIBC_2.27\x00GLIBCXX_3.4.23\x00')
            with self.assertRaisesRegex(ValueError, 'above 3.4.22'):
                mod.audit_elf(wheel)

    def test_vendored_metadata_is_not_package_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / 'demo-1.0-py3-none-any.whl'
            make_wheel(wheel)
            with zipfile.ZipFile(wheel, 'a') as archive:
                archive.writestr('demo/_vendor/other-9.dist-info/METADATA', 'Name: other\nVersion: 9\n')
            self.assertEqual(mod.wheel_identity(wheel), ('demo', '1.0'))

    def test_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('../escape', '/tmp/escape'):
                with self.assertRaises(ValueError):
                    mod.contained(Path(directory), name)


class BundleIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name)
        (self.bundle / 'python').mkdir()
        archive = self.bundle / 'python' / mod.PYTHON_ARCHIVE
        archive.write_bytes(b'fake-archive-for-unit-test')
        self.sha_patch = patch.object(mod, 'PYTHON_SHA', mod.digest(archive))
        self.sha_patch.start()
        self.addCleanup(self.sha_patch.stop)
        for group in mod.REQUIREMENTS:
            folder = self.bundle / 'wheels' / group
            folder.mkdir(parents=True)
            make_wheel(folder / 'demo-1.0-py3-none-any.whl')
        mod.build(self.bundle)

    def test_valid_round_trip(self):
        self.assertEqual(mod.verify(self.bundle)['python'], '3.12.15')

    def test_corruption_fails(self):
        wheel = next((self.bundle / 'wheels/wsl_tf').glob('*.whl'))
        wheel.write_bytes(wheel.read_bytes() + b'corrupted')
        with self.assertRaisesRegex(ValueError, 'integrity failed'):
            mod.verify(self.bundle)

    def test_extra_wheel_fails(self):
        (self.bundle / 'wheels/wsl_tf/extra.whl').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'Bundle files changed'):
            mod.verify(self.bundle)

    def test_lock_manifest_disagreement_fails_even_if_file_hash_updated(self):
        lock = self.bundle / 'wsl_tf.lock.txt'
        lock.write_text('unrelated==1.0\n')
        manifest_file = self.bundle / 'bundle.json'
        doc = json.loads(manifest_file.read_text())
        doc['files']['wsl_tf.lock.txt'] = {'sha256': mod.digest(lock), 'bytes': lock.stat().st_size}
        manifest_file.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, 'Install lock does not match'):
            mod.verify(self.bundle)

    def test_old_requirements_fail(self):
        file = self.bundle / 'bundle.json'
        doc = json.loads(file.read_text())
        doc['requirements']['wsl/requirements.txt'] = '0' * 64
        file.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, 'Requirements changed'):
            mod.verify(self.bundle)


if __name__ == '__main__':
    unittest.main()
