"""Integrity of stored waveform provenance and training-only INT8 calibration."""
import unittest
import tempfile
import hashlib
from pathlib import Path
import numpy as np
from day3.make_data import generate_arrays
from day3.numerics import features
from day3.export_checks import calibration_sample


class VibrationR2Tests(unittest.TestCase):
    def test_stored_waveforms_are_actual_network_rows(self):
        z, _ = generate_arrays(42, 10, 3)
        for split in ('train', 'val', 'test'):
            rows = z['example_rows_'+split]
            self.assertEqual(len(rows),8)
            self.assertEqual(len(set(rows)),8)
            np.testing.assert_array_equal(np.bincount(z['y_'+split][rows]),[2,2,2,2])
            actual=((features(z['raw_examples_'+split])-z['mean'])/z['std']).astype(np.float32)
            np.testing.assert_array_equal(actual,z['x_'+split][rows])

    def test_calibration_balanced_reproducible_and_train_only(self):
        z,_=generate_arrays(42,10,3)
        values,indices=calibration_sample(z)
        again,again_indices=calibration_sample(z)
        self.assertTrue(np.all(indices<len(z['x_train'])))
        np.testing.assert_array_equal(values,z['x_train'][indices])
        np.testing.assert_array_equal(np.bincount(z['y_train'][indices]),[18]*4)
        np.testing.assert_array_equal(indices,again_indices)
        z['x_val'][:]=np.nan;z['x_test'][:]=np.nan
        np.testing.assert_array_equal(calibration_sample(z)[0],again)

    def test_fft_amplitude_units_and_bins(self):
        from day3.visualize_data import amplitude_spectrum
        from day3.numerics import N, FS
        t=np.arange(N)/FS
        frequencies, amplitude=amplitude_spectrum(2*np.sin(2*np.pi*80*t))
        self.assertEqual(frequencies[1]-frequencies[0],2)
        self.assertEqual(frequencies[-1],1024)
        self.assertAlmostEqual(amplitude[40],2,places=4)

    def test_analysis_excludes_stale_model_or_test_data(self):
        from day3.analyze_results import collect_results
        from extra_common.util import write_json, sha256
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);(run/'data').mkdir();(run/'tflite').mkdir()
            x=np.zeros((4,256),np.float32);y=np.arange(4,dtype=np.int64)
            np.savez(run/'data/dataset.npz',x_test=x,y_test=y)
            model=run/'tflite/baseline_int8.tflite';model.write_bytes(b'current-model-bytes')
            row={'schema':'edge_benchmark_v1','backend':'tflite','variant':'baseline','precision':'int8',
                 'model':model.name,'model_sha256':sha256(model),
                 'input_data_sha256':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()}
            records=run/'results'
            write_json(records/'measurements.json',[row,dict(row,model_sha256='stale'),dict(row,input_data_sha256='stale')])
            accepted,excluded=collect_results([records],run)
            self.assertEqual(len(accepted),1)
            self.assertEqual(len(excluded),2)
            self.assertIn('model SHA',excluded[0]['reason'])
            self.assertIn('evaluation',excluded[1]['reason'])

    def test_calibration_normal_size(self):
        z={'x_train':np.arange(400*256,dtype=np.float32).reshape(400,256),'y_train':np.repeat(np.arange(4),100)}
        values,indices=calibration_sample(z)
        self.assertEqual(values.shape,(256,256))
        self.assertEqual(len(np.unique(indices)),256)
        np.testing.assert_array_equal(np.bincount(z['y_train'][indices]),[64]*4)


if __name__=='__main__':unittest.main()
