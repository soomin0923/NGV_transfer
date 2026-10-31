"""Run pretrained ImageNet CPU inference and optionally compare a custom classifier."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from extra_common.camera import camera_args, options
from extra_common.course import load_config
from extra_common.live_camera import LiveApp
from extra_common.webui import serve
from extra_common.util import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    camera_args(p)
    p.add_argument('--threads', type=int, default=None)
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--threshold', type=float, default=.65)
    p.add_argument('--resize-mode', choices=['stretch', 'crop'], default='stretch')
    p.add_argument('--custom-model')
    p.add_argument('--custom-labels')
    p.add_argument('--model', default=str(ROOT / 'day2_extra/artifacts/imagenet_mobilenetv3small_fp32.tflite'))
    p.add_argument('--labels', default=str(ROOT / 'day2_extra/artifacts/labels.txt'))
    a = p.parse_args()
    serve(LiveApp(options(a), 'imagenet', a.custom_model, a.custom_labels, a.threads, a.threshold,
                  resize_mode=a.resize_mode, imagenet_model=a.model, imagenet_labels=a.labels,
                  config=load_config(a.config)), a.port)


if __name__ == '__main__':
    main()
