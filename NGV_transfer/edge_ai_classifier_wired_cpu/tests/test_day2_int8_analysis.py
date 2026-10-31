"""Checks for calibration leakage and trustworthy classification comparisons."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from PIL import Image
from extra_common.calibration import load_training_calibration
from extra_common.util import sha256
from wsl.evaluate_tflite import classification_metrics
from wsl.analyze_results import ensure_comparable


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'dataset_split'
        self.records=[]
        for split,session in [('train','s01'),('val','s04'),('test','s05')]:
            for ci,label in enumerate(['alpha','beta']):
                folder=self.root/split/label;folder.mkdir(parents=True)
                for i in range(5):
                    f=folder/f'{session}_{i}.png'
                    rng=np.random.default_rng(ci*1000+len(self.records));Image.fromarray(rng.integers(0,256,(12,13,3),dtype=np.uint8)).save(f)
                    self.records.append({'split':split,'session':session,'class':label,'source':f'raw/{label}/{session}/{i}.png','sha256':sha256(f)})
        (self.root/'class_names.txt').write_text('alpha\nbeta\n')
        (self.root/'split_manifest.json').write_text(json.dumps(self.records))
    def tearDown(self):self.tmp.cleanup()
    def test_balanced_deterministic_and_shape(self):
        x,m=load_training_calibration(self.root/'train',(8,9),7)
        xx,mm=load_training_calibration(self.root/'train',(8,9),7)
        self.assertEqual(x.shape,(7,8,9,3));self.assertEqual(x.dtype,np.float32)
        self.assertEqual(m['class_counts'],{'alpha':4,'beta':3});self.assertTrue(m['split_manifest_verified'])
        np.testing.assert_array_equal(x,xx);self.assertEqual(m,mm)
    def test_heldout_refused(self):
        with self.assertRaisesRegex(ValueError,'train directory'):load_training_calibration(self.root/'test',max_images=4)
    def test_modified_train_refused(self):
        f=self.root/'train/alpha/s01_0.png';Image.new('RGB',(12,13),'red').save(f)
        with self.assertRaisesRegex(ValueError,'changed since'):load_training_calibration(self.root/'train',max_images=4)
    def test_heldout_hash_refused(self):
        rows=self.records.copy();rows.append({**rows[0],'split':'test'})
        (self.root/'split_manifest.json').write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError,'held-out'):load_training_calibration(self.root/'train',max_images=4)
    def test_missing_manifest_duplicate_refused(self):
        (self.root/'split_manifest.json').unlink()
        target=self.root/'train/alpha/s01_0.png';target.write_bytes((self.root/'test/alpha/s05_0.png').read_bytes())
        with self.assertRaisesRegex(ValueError,'shares identical'):load_training_calibration(self.root/'train',max_images=4)
    def test_budget_minimum(self):
        with self.assertRaisesRegex(ValueError,'at least'):load_training_calibration(self.root/'train',max_images=1)


class MetricsTests(unittest.TestCase):
    def test_zero_predicted_class(self):
        metrics=classification_metrics(np.array([[3,0],[2,0]]),['a','b'])
        self.assertEqual(metrics['accuracy'],.6);self.assertEqual(metrics['per_class'][1]['precision'],0.)
        self.assertAlmostEqual(metrics['macro_f1'],.375)
    def test_mixed_test_sets_rejected(self):
        base=dict(data_sha256='a',labels=['a','b'],samples=10,threads=4,batch=1,warmup=20,hostname='host',timing_scope='s')
        ensure_comparable([base,dict(base)])
        with self.assertRaisesRegex(ValueError,'data_sha256'):ensure_comparable([base,dict(base,data_sha256='b')])


if __name__=='__main__':unittest.main()
