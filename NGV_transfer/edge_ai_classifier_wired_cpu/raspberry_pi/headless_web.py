"""Collect configured classes or run CPU TFLite inference through the SSH web UI."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from extra_common.camera import camera_args, options
from extra_common.course import load_config
from extra_common.live_camera import LiveApp
from extra_common.webui import serve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    camera_args(p)
    p.add_argument('--mode', choices=['collect', 'infer'], required=True)
    p.add_argument('--model', help='TFLite model for infer mode')
    p.add_argument('--labels', help='Matching labels.txt produced with the model')
    p.add_argument('--threads', type=int, default=None, help='Default: course_config.json threads')
    p.add_argument('--threshold', type=float, default=.65)
    p.add_argument('--target', type=int, default=None, help='Photos per class/session; normally edit photos_per_session in course_config.json')
    p.add_argument('--interval', type=float, default=.5, help='Minimum seconds between saved images')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--output', default='dataset_raw')
    a = p.parse_args()
    course = load_config(a.config)
    serve(LiveApp(options(a), a.mode, a.model, a.labels, a.threads, a.threshold, a.target, a.interval,
                  output=a.output, config=course), a.port)


if __name__ == '__main__':
    main()
