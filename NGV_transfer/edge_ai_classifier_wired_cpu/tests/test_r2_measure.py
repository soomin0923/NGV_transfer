import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from extra_common.util import measure

class MeasureTests(unittest.TestCase):
    def test_confusion_comes_from_actual_predictions(self):
        class Runner:
            def predict(self,x):return np.eye(3,dtype=np.float32)[x[:,0].astype(int)]
        with tempfile.TemporaryDirectory() as d:
            model=Path(d)/'model';model.write_bytes(b'model')
            row=measure(Runner(),np.array([[0],[1],[1],[2]],np.float32),np.array([0,0,1,2]),model=model,backend='test',tag='unit',batch=2,warmup=0,repeats=1)
            self.assertEqual(row['confusion_matrix'],[[1,1,0],[0,1,0],[0,0,1]])
            self.assertEqual(row['accuracy'],.75)
            self.assertAlmostEqual(row['macro_f1'],7/9)
            self.assertEqual(row['samples'],4)

if __name__=='__main__':unittest.main()
