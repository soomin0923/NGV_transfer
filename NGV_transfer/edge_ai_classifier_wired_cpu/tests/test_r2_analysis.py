"""Guard scientific interpretation: confusion counts, model grouping, matched scoring."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
import numpy as np
from extra_common.analysis import confusion_counts, benchmark_groups
from day2_extra.analyze_comparison import grouped_summary
from day3_image.export_checks import require_validated, SCHEMA
from extra_common.util import sha256, write_json

class AnalysisTests(unittest.TestCase):
    def test_confusion_rows_are_truth_columns_prediction(self):
        cm=confusion_counts(np.array([0,0,1,2,2]),np.array([0,1,1,0,2]),3)
        np.testing.assert_array_equal(cm,[[1,1,0],[0,1,0],[1,0,1]])
        self.assertEqual(int(cm.sum()),5)

    def test_invalid_class_rejected(self):
        with self.assertRaises(ValueError):confusion_counts(np.array([0,3]),np.array([0,1]),3)

    def test_benchmark_devices_batch_threads_dataset_stay_separate(self):
        base=dict(schema='edge_benchmark_v1',tag='wsl',backend='litert',hostname='a',machine='x86_64',batch=1,threads=4,input_data_sha256='data',accuracy=.8,median_ms=2,p95_ms=3,bytes=100)
        rows=[base,dict(base,precision='int8'),dict(base,tag='pi'),dict(base,batch=64),dict(base,threads=2),dict(base,input_data_sha256='other'),dict(base,hostname='other')]
        self.assertEqual(len(benchmark_groups(rows)),6)

    def test_mapping_requires_same_frames_for_comparison(self):
        rows=[dict(image='a.jpg',truth='cup',condition='bright',custom_top1='cup',imagenet_top1='coffee_mug',imagenet_top5='coffee_mug|cup'),dict(image='b.jpg',truth='cup',condition='bright',custom_top1='bottle',imagenet_top1=''),dict(image='c.jpg',truth='other',condition='bright',custom_top1='other',imagenet_top1='cup')]
        result=grouped_summary(rows,{'cup':['coffee_mug']},['cup','bottle','other'])
        cup=next(r for r in result if r['truth']=='cup');other=next(r for r in result if r['truth']=='other')
        self.assertEqual(cup['custom_accuracy'],.5)
        self.assertEqual(cup['custom_accuracy_paired'],1)
        self.assertEqual(cup['imagenet_top1_mapped_accuracy_paired'],1)
        self.assertIsNone(other['imagenet_top1_mapped_accuracy_paired'])
        none=grouped_summary(rows,{},['cup','bottle','other'])
        self.assertTrue(all(r['imagenet_top1_mapped_accuracy_paired'] is None for r in none))

    def test_changed_model_files_are_not_pooled(self):
        base=dict(image='a.jpg',truth='cup',condition='bright',custom_top1='cup',imagenet_top1='coffee_mug',custom_model_sha256='one',imagenet_model_sha256='im')
        result=grouped_summary([base,dict(base,custom_model_sha256='two',custom_top1='bottle')],{'cup':['coffee_mug']},['cup','bottle'])
        self.assertEqual(len(result),2)
        self.assertEqual(sorted(r['custom_accuracy'] for r in result),[0,1])

    def test_validation_requires_six_tflite_but_four_onnx(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for kind,precisions in [('tflite',['fp32','dynamic','int8']),('onnx',['fp32','int8'])]:
                rows=[]
                for v in ['baseline','pruned']:
                    for p in precisions:
                        path=out/f'{v}_{p}.{kind}';path.write_bytes(b'unittest fixture')
                        rows.append(dict(schema=SCHEMA,variant=v,precision=p,validated=True,model_sha256=sha256(path)))
                write_json(out/f'{kind}_validation.json',rows)
                self.assertEqual(len(require_validated(out,kind)),len(rows))
            (out/'pruned_dynamic.tflite').unlink()
            with self.assertRaisesRegex(FileNotFoundError,'all 6'):require_validated(out,'tflite')
            self.assertEqual(len(require_validated(out,'onnx')),4)

if __name__=='__main__':unittest.main()
