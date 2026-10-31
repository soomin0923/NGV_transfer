"""Probe the connected USB camera, save a real frame and reusable camera settings."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
from extra_common.camera import Camera, camera_args, options, list_devices
from extra_common.util import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    camera_args(p)
    p.add_argument('--list-cameras', action='store_true', help='List Linux video devices without opening them')
    p.add_argument('--save-config', help='Save settings for --camera-config on capture/inference commands')
    p.add_argument('--sample', default='raspberry_pi/camera_sample.jpg')
    a = p.parse_args()
    if a.list_cameras:
        print('\n'.join(list_devices()) or 'No /dev/video* devices')
        return
    import cv2
    cam = Camera(options(a))
    try:
        for _ in range(3):
            cam.read()
        t = time.perf_counter()
        for _ in range(12):
            frame = cam.read()
        elapsed = time.perf_counter() - t
        Path(a.sample).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(a.sample, frame):
            raise RuntimeError('Cannot save camera sample')
        doc = {'requested':cam.requested, 'actual':cam.actual, 'sample':a.sample,
               'measured_read_fps':12 / max(elapsed, 1e-9)}
        if a.save_config:
            write_json(a.save_config, doc)
        print('PASS: actual frames read; sample saved:', a.sample)
        print('Measured read FPS (not inference FPS):', round(doc['measured_read_fps'], 2))
        if a.save_config:
            print('Saved configuration:', a.save_config)
    finally:
        cam.close()


if __name__ == '__main__':
    main()
