from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1]
manifest=root/'SOURCE_SHA256.json'
if not manifest.exists():raise SystemExit('Missing SOURCE_SHA256.json; use the complete final ZIP.')
doc=json.loads(manifest.read_text(encoding='utf-8'));errors=[]
for relative,expected in doc.items():
    p=root/relative
    if not p.is_file():errors.append('MISSING '+relative)
    elif hashlib.sha256(p.read_bytes()).hexdigest()!=expected:errors.append('CHANGED '+relative)
if errors:raise SystemExit('SOURCE VALIDATION FAILED:\n'+'\n'.join(errors))
print('PASS:',len(doc),'source files match this ZIP manifest. This is not a hardware test.')
