"""No model dependencies: verify offline data validation and held-out split integrity."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from day3_image.data import load_local, split_indices


class FashionLocalTests(unittest.TestCase):
    def setUp(self):
        self.y = np.repeat(np.arange(10), 6000)
        self.yt = np.repeat(np.arange(10), 1000)

    def test_default_balanced_heldout(self):
        parts = split_indices(self.y, self.yt)
        for name, count in [('train',12000),('val',3000),('test',3000)]:
            ids = parts[name]
            labels = self.yt if name == 'test' else self.y
            self.assertEqual(len(ids), count)
            self.assertEqual(len(np.unique(ids)), count)
            np.testing.assert_array_equal(np.bincount(labels[ids]), np.full(10,count//10))
        self.assertFalse(np.intersect1d(parts['train'], parts['val']).size)
        self.assertTrue(np.max(parts['test']) < len(self.yt))

    def test_seed_reproducibility(self):
        a,b,c=split_indices(self.y,self.yt),split_indices(self.y,self.yt),split_indices(self.y,self.yt,seed=7)
        for key in a:
            np.testing.assert_array_equal(a[key],b[key])
            self.assertFalse(np.array_equal(a[key],c[key]))

    def test_full_data(self):
        parts=split_indices(self.y,self.yt,full_data=True)
        self.assertEqual([len(parts[n]) for n in ['train','val','test']],[54000,6000,10000])
        self.assertEqual(len(set(parts['train'])|set(parts['val'])),60000)
        self.assertEqual(set(parts['test']),set(range(10000)))

    def test_invalid_counts(self):
        for value in [0,11,60000,True,12.0]:
            with self.assertRaises(ValueError):split_indices(self.y,self.yt,train_samples=value)
        with self.assertRaises(ValueError):split_indices(self.y,self.yt,test_samples=10010)

    def test_missing_local_file_is_actionable(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError,'never downloads'):
                load_local(Path(d)/'fashion-mnist.npz')

    def test_incomplete_npz_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.npz'
            np.savez(path,x_train=np.zeros((1,28,28),np.uint8),y_train=np.array([0]),x_test=np.zeros((1,28,28),np.uint8),y_test=np.array([0]))
            with self.assertRaisesRegex(ValueError,'complete official'):
                load_local(path)

    def test_wrong_npz_keys_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.npz';np.savez(path,images=np.zeros((1,28,28)))
            with self.assertRaisesRegex(ValueError,'expected keys'):
                load_local(path)


if __name__ == '__main__':unittest.main()
