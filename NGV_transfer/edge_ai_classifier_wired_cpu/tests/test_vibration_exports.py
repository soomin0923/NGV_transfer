"""Portable integrity checks for staged model publication (no TensorFlow required)."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from day3.export_checks import (compare, require_validated, start_export,
                               finish_export, SCHEMA, VARIANTS)
from extra_common.util import sha256


class VibrationExportTests(unittest.TestCase):
    def test_cpu_parity_rejects_real_drift(self):
        truth=np.array([[.1,.2,.3,.4]],np.float32)
        bad=truth.copy();bad[0,0]+=.05
        with self.assertRaises(ValueError):compare(bad,truth,'expected test failure',strict=True)
        self.assertFalse(compare(bad,truth,'quantized')['allclose'])

    def test_parity_shape_checked(self):
        with self.assertRaises(ValueError):compare(np.zeros((1,5)),np.zeros((1,5)),'bad')

    def test_publication_marker_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'published';stage=Path(d)/'stage';stage.mkdir()
            start_export(out)
            with self.assertRaisesRegex(RuntimeError,'incomplete'):
                require_validated(out,'tflite')
            names,rows=[],[]
            for variant in VARIANTS:
                for precision in ['fp32','dynamic','int8']:
                    name=f'{variant}_{precision}.tflite';path=stage/name
                    path.write_bytes((variant+precision).encode())
                    names.append(name);rows.append(dict(schema=SCHEMA,variant=variant,precision=precision,validated=True,model_sha256=sha256(path)))
            finish_export(out,stage,names,rows)
            self.assertEqual(len(require_validated(out,'tflite')),9)
            (out/names[0]).write_bytes(b'replaced-after-validation')
            with self.assertRaisesRegex(ValueError,'Unvalidated/replaced'):
                require_validated(out,'tflite')


if __name__=='__main__':unittest.main()
