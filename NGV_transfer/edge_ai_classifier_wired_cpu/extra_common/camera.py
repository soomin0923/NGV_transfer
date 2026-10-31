"""USB/V4L2 capture with actual-frame sizing and conservative auto probing.

Zero width/height/FPS and FOURCC=AUTO leave negotiation to the device. Auto
mode tries the OS defaults first, then common MJPG/YUYV modes only if those
provide no readable frames. Camera driver, power and supported formats still
matter; this does not promise support for every USB camera.
"""
from __future__ import annotations
import glob
import json
import time
from pathlib import Path
import numpy as np


def camera_args(parser):
    parser.add_argument('--config', default=None, help='Course JSON; default: course_config.json at project root')
    parser.add_argument('--camera', default=None, help='auto, index, /dev/videoN, or persistent /dev/v4l/by-id/...')
    parser.add_argument('--width', type=int, default=None, help='Requested width; 0 uses camera default (set height too)')
    parser.add_argument('--height', type=int, default=None, help='Requested height; 0 uses camera default')
    parser.add_argument('--fps', type=float, default=None, help='Requested FPS; 0 uses camera default')
    parser.add_argument('--fourcc', default=None, help='AUTO, MJPG, YUYV, or another supported four-character format')
    parser.add_argument('--camera-config', default=None, help='JSON written by camera_check.py --save-config; overrides course camera settings')


def options(args):
    from extra_common.course import load_config
    base = dict(load_config(getattr(args, 'config', None))['camera'])
    if getattr(args, 'camera_config', None):
        doc = json.loads(Path(args.camera_config).read_text(encoding='utf-8-sig'))
        base.update(doc.get('requested', doc))
    for key in ('camera', 'width', 'height', 'fps', 'fourcc'):
        value = getattr(args, key, None)
        if value is not None:
            base[key] = value
    base = {key: base[key] for key in ('camera', 'width', 'height', 'fps', 'fourcc')}
    base['camera'] = str(base['camera'])
    base['fourcc'] = str(base['fourcc']).upper()
    if base['width'] < 0 or base['height'] < 0 or base['fps'] < 0:
        raise ValueError('Negative camera size/FPS')
    if bool(base['width']) != bool(base['height']):
        raise ValueError('Set both --width and --height, or neither')
    if base['fourcc'] != 'AUTO' and len(base['fourcc']) != 4:
        raise ValueError('FOURCC must be AUTO or four characters')
    return base


def list_devices():
    """Prefer persistent USB links and omit duplicate physical video nodes."""
    seen, out = set(), []
    for name in sorted(glob.glob('/dev/v4l/by-id/*')) + sorted(glob.glob('/dev/video*')):
        real = str(Path(name).resolve())
        if real not in seen:
            seen.add(real)
            out.append(name)
    return out


def roi_bounds(shape, fraction=.70):
    if not 0 < fraction <= 1:
        raise ValueError('ROI fraction must be in (0,1]')
    height, width = shape[:2]
    if height < 1 or width < 1:
        raise ValueError('Empty camera frame')
    side = max(1, int(min(height, width) * fraction))
    x, y = (width - side) // 2, (height - side) // 2
    return x, y, x + side, y + side


def crop_roi(frame, fraction=.70):
    x1, y1, x2, y2 = roi_bounds(frame.shape, fraction)
    return frame[y1:y2, x1:x2].copy()


def model_input(bgr, height, width, resize_mode='stretch'):
    import cv2
    if resize_mode == 'crop':
        h, w = bgr.shape[:2]
        target = width / height
        if w / h > target:
            new_w = max(1, round(h * target))
            start = (w - new_w) // 2
            bgr = bgr[:, start:start + new_w]
        else:
            new_h = max(1, round(w / target))
            start = (h - new_h) // 2
            bgr = bgr[start:start + new_h, :]
    elif resize_mode != 'stretch':
        raise ValueError(resize_mode)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)[None]


def _bgr(frame):
    import cv2
    if frame is None or not frame.size:
        raise RuntimeError('Empty camera frame')
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f'Unsupported decoded camera frame shape: {frame.shape}')
    return frame


class Camera:
    def __init__(self, config):
        import cv2
        cfg = dict(config)
        source = str(cfg['camera'])
        candidates = list_devices() if source == 'auto' else [source]
        if not candidates and source == 'auto':
            candidates = ['0']
        self.cap = None
        failures = []
        automatic_format = not cfg['width'] and not cfg['height'] and not cfg['fps'] and cfg['fourcc'].upper() == 'AUTO'
        attempts = [cfg]
        if automatic_format:
            attempts += [{**cfg, 'width':640, 'height':480, 'fps':15, 'fourcc':fmt} for fmt in ('MJPG', 'YUYV')]
        for name in candidates:
            src = int(name) if str(name).isdigit() else name
            for attempt in attempts:
                cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    failures.append(f'{name}: open failed')
                    break
                if attempt['fourcc'].upper() != 'AUTO':
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*attempt['fourcc']))
                if attempt['width']:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, attempt['width'])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, attempt['height'])
                if attempt['fps']:
                    cap.set(cv2.CAP_PROP_FPS, attempt['fps'])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                frame = None
                for _ in range(10):
                    ok, img = cap.read()
                    if ok and img is not None and img.size:
                        frame = _bgr(img)
                        break
                    time.sleep(.05)
                if frame is None:
                    cap.release()
                    failures.append(f"{name}: no frames ({attempt['fourcc']} {attempt['width']}x{attempt['height']})")
                    continue
                self.cap, self.first, self.source = cap, frame, name
                self.requested = {**attempt, 'camera':str(name)}
                code = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc = ''.join(chr((code >> (8 * i)) & 255) for i in range(4)).strip('\x00')
                reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
                self.actual = {'device':str(name), 'width':int(frame.shape[1]), 'height':int(frame.shape[0]),
                               'reported_fps':reported_fps if np.isfinite(reported_fps) else 0.0, 'fourcc':fourcc}
                if attempt is not cfg:
                    print('CAMERA: default mode had no readable frames; selected fallback mode.', flush=True)
                print('CAMERA requested:', self.requested, flush=True)
                print('CAMERA actual:', self.actual, flush=True)
                if attempt['width'] and (frame.shape[1] != attempt['width'] or frame.shape[0] != attempt['height']):
                    print('WARNING: camera negotiated a different resolution. Actual frame dimensions are used.', flush=True)
                break
            if self.cap is not None:
                break
        if self.cap is None:
            raise RuntimeError('No working video stream. Check v4l2-ctl formats, USB power/permissions and other camera processes. ' + '; '.join(failures))

    def read(self):
        if self.first is not None:
            frame, self.first = self.first, None
        else:
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError('Camera frame read failed. Check power, device and supported format.')
            frame = _bgr(frame)
        self.actual['width'], self.actual['height'] = int(frame.shape[1]), int(frame.shape[0])
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
