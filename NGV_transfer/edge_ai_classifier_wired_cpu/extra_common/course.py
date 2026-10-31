"""Portable course settings. Paths are relative to this code, never a username."""
from __future__ import annotations
import copy, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'labels': ['bottle', 'cup', 'other', 'remote', 'wallet'],
    'photos_per_session': 40,
    'sessions': ['s01', 's02', 's03', 's04', 's05'],
    'train_sessions': ['s01', 's02', 's03'],
    'val_sessions': ['s04'], 'test_sessions': ['s05'], 'threads': 4,
    'camera': {'camera': 'auto', 'width': 0, 'height': 0, 'fps': 0, 'fourcc': 'AUTO'},
}

def validate_config(cfg):
    names = cfg['labels']
    if not isinstance(names, list) or len(names) < 2 or len(names) != len(set(names)):
        raise ValueError('Use at least two distinct labels.')
    for name in names:
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}', name):
            raise ValueError('Labels: ASCII letters first, then letters/numbers/_/-, max 40 characters. Example: red_cup')
        if name.lower() in {'con','prn','aux','nul',*[f'com{i}' for i in range(1,10)],*[f'lpt{i}' for i in range(1,10)],'out_of_scope'}:
            raise ValueError('Reserved label: '+name)
    if len({s.lower() for s in names}) != len(names):
        raise ValueError('Labels must differ ignoring case (Windows compatibility).')
    if not isinstance(cfg['photos_per_session'], int) or cfg['photos_per_session'] < 2:
        raise ValueError('photos_per_session must be >= 2')
    if not isinstance(cfg['threads'], int) or not 1 <= cfg['threads'] <= 64:
        raise ValueError('threads must be 1..64')
    sessions = cfg['sessions']
    groups = [cfg[k] for k in ['train_sessions','val_sessions','test_sessions']]
    flat = [s for group in groups for s in group]
    if any(not group for group in groups) or len(flat) != len(set(flat)) or set(flat) != set(sessions):
        raise ValueError('Train/val/test sessions must be nonempty, disjoint and cover sessions.')
    if any(not re.fullmatch(r's[0-9]{2}', s) for s in sessions):
        raise ValueError('Session names must be s01, s02, ...')
    return cfg

def load_config(path=None):
    path = Path(path) if path else ROOT/'course_config.json'
    cfg = copy.deepcopy(DEFAULTS)
    if path.exists():
        doc = json.loads(path.read_text(encoding='utf-8-sig'))
        camera = doc.pop('camera', {})
        cfg.update(doc); cfg['camera'].update(camera)
    elif path != ROOT/'course_config.json':
        raise FileNotFoundError(path)
    return validate_config(cfg)
