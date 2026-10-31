"""Apply R6 code from this extracted release to an existing CPU project.

Run this script from a separate extracted R6 folder. User data/configuration,
models, virtual environments and offline assets remain in the target.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil


def main():
    source=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',required=True,type=Path)
    a=p.parse_args();target=a.target.expanduser().resolve()
    def below(path, parent):
        try: path.relative_to(parent); return True
        except ValueError: return False
    if target==source or below(source,target) or below(target,source):
        p.error('Extract R6 beside/outside the existing target, not inside it.')
    if not (target/'wsl/train_export.py').is_file() or not (target/'course_config.json').is_file():
        p.error('Target must be the existing edge_ai_classifier_wired_cpu project root.')
    manifest=json.loads((source/'SOURCE_SHA256.json').read_text())
    names=list(manifest)+['VERSION.txt','README_KO.md','VALIDATION_CPU.txt','NGV_CPU_Simple_Guide_20261007_R6.html','SOURCE_SHA256.json']
    names=list(dict.fromkeys(names))
    for name in names:
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or not (source/rel).is_file():raise ValueError('Invalid/missing release file: '+name)
        if name in manifest and hashlib.sha256((source/rel).read_bytes()).hexdigest()!=manifest[name]:raise ValueError('Release checksum mismatch: '+name)
        if (target/rel).is_symlink():raise ValueError('Refusing to replace symlink: '+name)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    backup=target/'backups'/('before_R6_'+stamp);backup.mkdir(parents=True)
    added=[];copied=[]
    for name in names:
        dst=target/name
        if dst.exists():
            (backup/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,backup/name)
        else:added.append(name)
        dst.parent.mkdir(parents=True,exist_ok=True)
        tmp=dst.with_name(dst.name+'.r6_update_tmp');shutil.copy2(source/name,tmp);tmp.replace(dst);copied.append(name)
    (backup/'update_record.json').write_text(json.dumps({'source':str(source),'target':str(target),'updated':copied,'new_files':added},indent=2))
    for name,digest in manifest.items():
        if hashlib.sha256((target/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Post-update checksum mismatch: '+name)
    print('PASS: updated',len(copied),'code/document files. Backup:',backup)
    print('Preserved: course_config.json, camera_config.json, datasets, models, offline assets and virtual environments.')
    print('R6 setup uses a verified private Python runtime. Run bash setup_laptop.sh with internet access. The installer downloads its own packages and prepares offline/raw locally. Existing incompatible virtual environments are not silently deleted. Training data/models need not be recreated.')


if __name__=='__main__':main()
