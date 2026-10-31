from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import unittest
import numpy as np
from day3.numerics import magnitude_mask,prune_mlp,mlp_numpy,parameters,features,signal
class Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(1);self.w=[rng.normal(size=s).astype(np.float32) for s in [(256,512),(512,),(512,256),(256,),(256,4),(4,)]]
    def test_counts(self):self.assertEqual(parameters(self.w),263940)
    def test_structured_counts(self):self.assertEqual(parameters(prune_mlp(self.w,.5)[0]),99204)
    def test_mask(self):self.assertEqual(np.count_nonzero(magnitude_mask(self.w[0],.5)==0),self.w[0].size//2)
    def test_output_shape(self):self.assertEqual(mlp_numpy(np.zeros((3,256)),self.w).shape,(3,4))
    def test_probabilities(self):np.testing.assert_allclose(mlp_numpy(np.zeros((3,256)),self.w).sum(1),1,atol=1e-6)
    def test_keep_all(self):
        w,_=prune_mlp(self.w,1)
        for x,y in zip(self.w,w):np.testing.assert_array_equal(x,y)
    def test_signal(self):self.assertEqual(signal(0,np.random.default_rng(1)).shape,(1024,))
    def test_features(self):self.assertEqual(features(np.zeros(1024)).shape,(256,))
if __name__=='__main__':unittest.main()
