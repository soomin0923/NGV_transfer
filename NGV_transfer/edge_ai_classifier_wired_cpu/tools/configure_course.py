"""Set collection labels/counts once; copy course_config.json to your Pi."""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extra_common.course import ROOT, load_config, validate_config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', help='Comma-separated safe folder labels, in model output order')
    p.add_argument('--photos-per-session', type=int)
    p.add_argument('--threads', type=int)
    p.add_argument('--output', default=str(ROOT/'course_config.json'))
    a=p.parse_args(); cfg=load_config(a.output if Path(a.output).exists() else None)
    if a.labels is None and a.photos_per_session is None and a.threads is None:
        a.labels=input('Labels ['+','.join(cfg['labels'])+']: ').strip() or ','.join(cfg['labels'])
        a.photos_per_session=int(input(f"Photos/session [{cfg['photos_per_session']}]: ").strip() or cfg['photos_per_session'])
        a.threads=int(input(f"CPU threads [{cfg['threads']}]: ").strip() or cfg['threads'])
    if a.labels is not None: cfg['labels']=[s.strip() for s in a.labels.split(',')]
    if a.photos_per_session is not None: cfg['photos_per_session']=a.photos_per_session
    if a.threads is not None: cfg['threads']=a.threads
    validate_config(cfg)
    dst=Path(a.output); dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('SAVED',dst); print('Labels in output order:',cfg['labels'])
    print('Total images:',len(cfg['labels'])*len(cfg['sessions'])*cfg['photos_per_session'])
    print('Copy this same course_config.json to the Pi before capture. Do not change labels midway through a run.')

if __name__=='__main__': main()
