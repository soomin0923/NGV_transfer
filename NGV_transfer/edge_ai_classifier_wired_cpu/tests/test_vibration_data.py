"""CPU/NumPy checks; these do not claim TensorFlow or physical hardware execution."""
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from day3.make_data import generate_arrays
from day3.numerics import signal, features
from extra_common.util import measure
from day3_image.export_checks import compare, enforce


class VibrationDataTests(unittest.TestCase):
    def setUp(self):
        self.arrays, self.description = generate_arrays(42, 10, 3)

    def test_default_size(self):
        defaults = inspect.signature(generate_arrays).parameters
        self.assertEqual(defaults['sessions_per_class'].default, 150)
        self.assertEqual(defaults['windows_per_session'].default, 40)
        self.assertEqual(4 * 150 * 40, 24000)

    def test_counts_and_shapes(self):
        for split, n in [('train', 72), ('val', 24), ('test', 24)]:
            self.assertEqual(self.arrays['x_' + split].shape, (n, 256))
            np.testing.assert_array_equal(np.bincount(self.arrays['y_' + split]), [n//4]*4)
        self.assertEqual(self.description['train_val_test'], [72, 24, 24])

    def test_disjoint_sessions(self):
        sets = [set(self.arrays['session_' + s]) for s in ['train', 'val', 'test']]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(len(set.union(*sets)), 40)

    def test_dtypes_and_finite(self):
        for split in ['train', 'val', 'test']:
            self.assertEqual(self.arrays['x_' + split].dtype, np.float32)
            self.assertEqual(self.arrays['y_' + split].dtype, np.int64)
            self.assertTrue(np.isfinite(self.arrays['x_' + split]).all())
        self.assertTrue((self.arrays['std'] > 0).all())

    def test_training_normalization(self):
        np.testing.assert_allclose(self.arrays['x_train'].mean(0), 0, atol=3e-5)
        np.testing.assert_allclose(self.arrays['x_train'].std(0), 1, atol=3e-5)
        self.assertFalse(np.allclose(self.arrays['x_val'].mean(0), 0, atol=1e-3))

    def test_reproducibility_and_seed_effect(self):
        same, _ = generate_arrays(42, 10, 3)
        different, _ = generate_arrays(43, 10, 3)
        for key in self.arrays:
            np.testing.assert_array_equal(self.arrays[key], same[key])
        self.assertFalse(np.array_equal(self.arrays['x_test'], different['x_test']))

    def test_metadata(self):
        rows = self.description['sessions']
        self.assertEqual(len(rows), 40)
        self.assertTrue(self.description['synthetic_only'])
        self.assertEqual(self.description['normalization'], 'train only')
        self.assertEqual(sum(r['windows'] for r in rows), 120)

    def test_invalid_arguments(self):
        for args in [(-1, 10, 3), (42, 0, 3), (42, 11, 3), (42, 10, 0), (42, 10, -1),
                     (42, 10.0, 3), (True, 10, 3)]:
            with self.assertRaises(ValueError):
                generate_arrays(*args)

    def test_original_recipe_array_equality(self):
        # Independent transcription of the uploaded 30-session/40-window loop.
        rng = np.random.default_rng(42)
        data = {s: [] for s in ['train', 'val', 'test']}
        ys = {s: [] for s in data}
        ids = {s: [] for s in data}
        for kind in range(4):
            for session in range(30):
                split = 'train' if session < 18 else 'val' if session < 24 else 'test'
                freq = rng.uniform(34, 46); gain = rng.uniform(.8, 1.2); noise = rng.uniform(.04, .12)
                for _ in range(40):
                    data[split].append(features(signal(kind, rng, freq, gain, noise)))
                    ys[split].append(kind); ids[split].append(kind * 30 + session)
        mean = np.mean(data['train'], axis=0)
        std = np.maximum(np.std(data['train'], axis=0), 1e-5)
        actual, _ = generate_arrays(42, 30, 40)
        np.testing.assert_array_equal(actual['mean'], mean)
        np.testing.assert_array_equal(actual['std'], std)
        for split in data:
            np.testing.assert_array_equal(actual['x_' + split], ((np.stack(data[split])-mean)/std).astype(np.float32))
            np.testing.assert_array_equal(actual['y_' + split], ys[split])
            np.testing.assert_array_equal(actual['session_' + split], ids[split])

    def test_cli_output_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'data'
            command = [sys.executable, str(ROOT/'day3/make_data.py'), '--sessions-per-class', '5',
                       '--windows-per-session', '2', '--output', str(out)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            before = (out/'dataset.npz').read_bytes()
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn('STOP', second.stdout + second.stderr)
            self.assertEqual(before, (out/'dataset.npz').read_bytes())


class ScoreIntegrityTests(unittest.TestCase):
    def test_actual_score_can_be_zero_half_or_one(self):
        # Inject deliberately right/wrong predictions into the real score aggregator.
        x = np.eye(4, dtype=np.float32); y = np.arange(4)
        class Runner:
            def __init__(self, predictions): self.predictions = predictions
            def predict(self, batch): return np.eye(4)[self.predictions[np.argmax(batch, axis=1)]]
        with tempfile.TemporaryDirectory() as td:
            model = Path(td)/'not_a_real_model.bin'; model.write_bytes(b'unit-test-only')
            for pred, expected in [(np.arange(4), 1.), (np.array([1,0,2,3]), .5), (np.array([1,2,3,0]), 0.)]:
                result = measure(Runner(pred), x, y, model=model, backend='unit-test', tag='unit-test',
                                 batch=3, warmup=0, repeats=1)
                self.assertEqual(result['accuracy'], expected)
                self.assertEqual(result['samples'], 4)

    def test_top1_agreement_is_not_accuracy(self):
        # Identical outputs imply conversion agreement, even when the label is different.
        prediction = np.eye(10, dtype=np.float32)[[0, 0]]
        metrics = compare(prediction, prediction)
        self.assertEqual(metrics['top1_agreement'], 1.)
        self.assertEqual(np.mean(prediction.argmax(1) == np.array([1, 2])), 0.)

    def test_fp32_drift_is_rejected(self):
        original = np.full((2, 10), .1, np.float32)
        candidate = original.copy(); candidate[0, 0] += .02; candidate[0, 1] -= .02
        stats = compare(candidate, original)
        with self.assertRaises(ValueError):
            enforce(stats, 'intentional unit-test mismatch')


if __name__ == '__main__':
    unittest.main()
