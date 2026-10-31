
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,tempfile,unittest
from unittest.mock import patch
import numpy as np
from day3.numerics import *
from day3_image.numerics import prune_cnn,flatten_channel_indices
from extra_common.camera import options,camera_args,roi_bounds,crop_roi,model_input,Camera
from extra_common.runtime import quantize_input,dequantize_output
from extra_common.util import labels,finite,write_json,read_json
from day2_extra.summarize_comparison import summarize, MAPPING

class CoreTests(unittest.TestCase):
    def test_mlp_counts(self):
        w=[np.ones(s,np.float32) for s in [(256,512),(512,),(512,256),(256,),(256,4),(4,)]]
        self.assertEqual(parameters(w),263940);self.assertEqual(parameters(prune_mlp(w,.5)[0]),99204)
    def test_mlp_kept_neurons(self):
        rng=np.random.default_rng(42);w=[rng.normal(size=s) for s in [(256,8),(8,),(8,6),(6,),(6,4),(4,)]]
        small,(i,j)=prune_mlp(w,.5)
        ref=[w[0].copy(),w[1].copy(),w[2].copy(),w[3].copy(),w[4].copy(),w[5].copy()]
        ref[2][np.setdiff1d(np.arange(8),i)]=0;ref[4][np.setdiff1d(np.arange(6),j)]=0
        # Removed first-layer activations do not contribute to retained second layer.
        x=rng.normal(size=(3,256));np.testing.assert_allclose(mlp_numpy(x,small),mlp_numpy(x,ref),atol=1e-8)
    def test_masks_exact_stable(self):
        w=np.ones((8,8));mask=magnitude_mask(w,.5);self.assertEqual(np.count_nonzero(mask==0),32);self.assertTrue(np.all(mask.ravel()[:32]==0))
    def test_invalid_mask(self):
        for v in [-.1,1.1]:
            with self.assertRaises(ValueError):magnitude_mask(np.ones(4),v)
    def test_signal_repeatable(self):np.testing.assert_array_equal(signal(3,np.random.default_rng(42)),signal(3,np.random.default_rng(42)))
    def test_features(self):
        y=features(np.zeros((2,1024)));self.assertEqual(y.shape,(2,256));np.testing.assert_array_equal(y,0)
    def test_flatten_mapping(self):
        x=np.arange(2*3*4).reshape(2,3,4);keep=[1,3];np.testing.assert_array_equal(x.ravel()[flatten_channel_indices(2,3,4,keep)],x[:,:,keep].ravel())
    def test_cnn_shapes(self):
        w=[np.ones(s,np.float32) for s in [(3,3,1,32),(32,),(3,3,32,64),(64,),(7*7*64,128),(128,),(128,10),(10,)]]
        q,_=prune_cnn(w,.5);self.assertEqual([v.shape for v in q],[(3,3,1,16),(16,),(3,3,16,32),(32,),(7*7*32,64),(64,),(64,10),(10,)])
    def test_cnn_keep_all(self):
        rng=np.random.default_rng(5);w=[rng.normal(size=s) for s in [(3,3,1,4),(4,),(3,3,4,8),(8,),(7*7*8,16),(16,),(16,10),(10,)]]
        q,_=prune_cnn(w,1)
        for x,y in zip(w,q):np.testing.assert_array_equal(x,y)
    def test_camera_default_native(self):
        p=argparse.ArgumentParser();camera_args(p);a=p.parse_args([]);self.assertEqual(options(a),{'camera':'auto','width':0,'height':0,'fps':0,'fourcc':'AUTO'})
    def test_camera_config_override(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'camera.json';write_json(f,{'requested':{'camera':'/dev/video2','width':640,'height':480,'fps':15,'fourcc':'YUYV'}})
            p=argparse.ArgumentParser();camera_args(p);a=p.parse_args(['--camera-config',str(f),'--fps','10']);self.assertEqual(options(a)['fps'],10);self.assertEqual(options(a)['camera'],'/dev/video2')
    def test_camera_bad_dims(self):
        p=argparse.ArgumentParser();camera_args(p)
        with self.assertRaises(ValueError):options(p.parse_args(['--width','640']))
    def test_roi_any_resolution(self):
        for h,w in [(480,640),(720,1280),(1080,1920),(333,555),(16,24)]:
            x1,y1,x2,y2=roi_bounds((h,w,3));self.assertEqual(x2-x1,y2-y1);self.assertTrue(0<=x1<x2<=w and 0<=y1<y2<=h)
    def test_clean_roi_copy(self):
        f=np.ones((100,160,3),np.uint8);crop=crop_roi(f);crop[:]=0;self.assertTrue(np.all(f==1))
    def test_rgb_not_double_scaled(self):
        f=np.zeros((15,20,3),np.uint8);f[:,:,0]=255;x=model_input(f,224,224);self.assertEqual(x.shape,(1,224,224,3));self.assertEqual(x.dtype,np.float32);self.assertTrue(np.all(x[:,:,:,2]==255));self.assertTrue(np.all(x[:,:,:,0]==0))
    def test_quant_scale_zero_point(self):
        detail={'dtype':np.int8,'quantization':(.5,-3)};q=quantize_input(np.array([-100,0,1,100]),detail);np.testing.assert_array_equal(q,[-128,-3,-1,127])
    def test_dequant(self):np.testing.assert_allclose(dequantize_output(np.array([-3,-1],np.int8),{'dtype':np.int8,'quantization':(.5,-3)}),[0,1])
    def test_bad_quant_scale(self):
        with self.assertRaises(ValueError):quantize_input(np.ones(1),{'dtype':np.int8,'quantization':(0,0)})
    def test_nonfinite_input(self):
        with self.assertRaises(ValueError):quantize_input(np.array([np.nan]),{'dtype':np.float32})
    def test_nonfinite_output(self):
        with self.assertRaises(ValueError):dequantize_output(np.array([np.inf]),{'dtype':np.float32})
    def test_duplicate_imagenet_names_allowed_explicitly(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'labels.txt';f.write_text('crane\ncrane\n')
            self.assertEqual(labels(f,2,allow_duplicates=True),['crane','crane'])
            with self.assertRaises(ValueError):labels(f)
    def test_comparison_mapping(self):
        rows=[{'truth':'cup','condition':'a','custom_top1':'cup','imagenet_top1':'coffee_mug','imagenet_top5':'coffee_mug|vase'},{'truth':'out_of_scope','condition':'a'}]
        out=summarize(rows,mapping=MAPPING);self.assertEqual(len(out),1);self.assertEqual(out[0]['custom_accuracy_paired'],1);self.assertEqual(out[0]['imagenet_top1_mapped_accuracy'],1)
    def test_json_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):write_json(Path(d)/'x.json',{'x':float('nan')})
    def test_camera_native_no_forced_properties(self):
        import cv2
        class Fake:
            def __init__(self):self.settings=[]
            def isOpened(self):return True
            def set(self,*x):self.settings.append(x);return True
            def get(self,key):return 15 if key==cv2.CAP_PROP_FPS else 0
            def read(self):return True,np.zeros((333,555,3),np.uint8)
            def release(self):pass
        fake=Fake()
        with patch('cv2.VideoCapture',return_value=fake):
            cam=Camera({'camera':'2','width':0,'height':0,'fps':0,'fourcc':'AUTO'});self.assertEqual(cam.actual['width'],555);self.assertEqual(cam.actual['height'],333);cam.close()
        for prop,value in fake.settings:self.assertNotIn(prop,[cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS,cv2.CAP_PROP_FOURCC])

class CollectionTests(unittest.TestCase):
    def test_collect_target_clean_images_and_selection(self):
        import time,cv2
        from extra_common.live_camera import LiveApp
        class FakeCamera:
            actual={'width':123,'height':77,'device':'mock'}
            def __init__(self,*args):pass
            def read(self):return np.full((77,123,3),80,np.uint8)
            def close(self):pass
        with tempfile.TemporaryDirectory() as d,patch('extra_common.live_camera.Camera',FakeCamera):
            app=LiveApp({},mode='collect',target=3,interval=.01,output=d)
            try:
                app.action('/select',{'class':'cup','session':'s01'});app.action('/start',{})
                deadline=time.monotonic()+3
                while app.count()<3 and time.monotonic()<deadline:time.sleep(.03)
                self.assertEqual(app.count(),3);app.action('/start',{});time.sleep(.06);self.assertEqual(app.count(),3)
                for p in (Path(d)/'cup/s01').glob('*.jpg'):
                    image=cv2.imread(str(p));self.assertEqual(image.shape[0],image.shape[1]);self.assertTrue(np.max(np.abs(image.astype(int)-80))<=1)
                with self.assertRaises(ValueError):app.action('/select',{'class':'../../outside','session':'s01'})
            finally:app.close()

if __name__=='__main__':unittest.main()
