"""Local internet-download cache integrity and ABI checks. Uses the Python standard library only."""
from __future__ import annotations
import argparse
import email.parser
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ARCHIVE = 'cpython-3.12.15+20261003-x86_64-unknown-linux-gnu-install_only.tar.gz'
PYTHON_SHA = 'f937814031eab4698ca6d07ec606ede1825768f3f3e99af76d9db3900bee03c5'
REQUIREMENTS = {'wsl_tf': 'wsl/requirements.txt', 'wsl_onnx': 'day3/requirements_tools.txt'}
SHA_RE = re.compile(r'[0-9a-f]{64}')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def contained(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts or not value.parts:
        raise ValueError(f'Unsafe bundle path: {relative}')
    candidate = root / value
    if not candidate.resolve().is_relative_to(root.resolve()) or candidate.is_symlink():
        raise ValueError(f'Bundle symlinks/outside paths are forbidden: {relative}')
    return candidate


def compatible_wheel(filename: str) -> bool:
    """Require at least one CPython 3.12-compatible, glibc<=2.27 wheel tag."""
    if not filename.endswith('.whl'):
        return False
    try:
        _, py, abi, platform = filename[:-4].rsplit('-', 3)
    except ValueError:
        return False
    py_tags, abi_tags, platforms = py.split('.'), abi.split('.'), platform.split('.')
    abi_ok = 'cp312' in abi_tags or 'none' in abi_tags or 'abi3' in abi_tags
    py_ok = any(tag in ('cp312', 'py3', 'py312') for tag in py_tags)
    if 'abi3' in abi_tags:
        py_ok |= any(re.fullmatch(r'cp3\d+', tag) and int(tag[3:]) <= 12 for tag in py_tags)
    if not (abi_ok and py_ok):
        return False
    for tag in platforms:
        if tag == 'any':
            return True
        if tag in ('manylinux1_x86_64', 'manylinux2010_x86_64', 'manylinux2014_x86_64'):
            return True
        match = re.fullmatch(r'manylinux_2_(\d+)_x86_64', tag)
        if match and 5 <= int(match[1]) <= 27:
            return True
    return False


def audit_elf(wheel: Path) -> dict:
    """Check required version strings in all ELF payloads, with bounded memory."""
    versions = {'GLIBC': set(), 'GLIBCXX': set()}
    elf_count = 0
    pattern = re.compile(rb'(GLIBCXX|GLIBC)_(\d+(?:\.\d+)+)')
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            with archive.open(info) as stream:
                header = stream.read(4)
                if header != b'\x7fELF':
                    continue
                elf_count += 1
                carry = header
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    data = carry + block
                    for family, version in pattern.findall(data):
                        versions[family.decode()].add(tuple(map(int, version.split(b'.'))))
                    carry = data[-64:]
    maxima = {key: max(values) if values else () for key, values in versions.items()}
    if maxima['GLIBC'] > (2, 27):
        raise ValueError(f'{wheel.name} requires GLIBC {maxima["GLIBC"]}, above the 2.27 classroom baseline')
    if maxima['GLIBCXX'] > (3, 4, 22):
        raise ValueError(f'{wheel.name} requires GLIBCXX {maxima["GLIBCXX"]}, above 3.4.22')
    return {'elf_payloads': elf_count, **{key: '.'.join(map(str, value)) for key, value in maxima.items()}}


def wheel_identity(wheel: Path) -> tuple[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        metadata = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA') and n.count('/') == 1]
        if len(metadata) != 1:
            raise ValueError(f'Unexpected metadata count: {wheel.name}')
        parsed = email.parser.BytesParser().parsebytes(archive.read(metadata[0]))
    name, version = parsed['Name'], parsed['Version']
    if not name or not version or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or not re.fullmatch(r'[A-Za-z0-9_.+!-]+', version):
        raise ValueError(f'Invalid package metadata: {wheel.name}')
    return name, version


def build(bundle: Path) -> dict:
    archive = bundle / 'python' / PYTHON_ARCHIVE
    if digest(archive) != PYTHON_SHA:
        raise ValueError('Official CPython archive SHA256 mismatch')
    document = {'schema': 'ngv_wsl_bundle_v1', 'python': '3.12.15', 'architecture': 'x86_64',
                'glibc_min': '2.27', 'glibcxx_max': '3.4.22', 'python_sha256': PYTHON_SHA,
                'requirements': {path: digest(ROOT / path) for path in REQUIREMENTS.values()},
                'groups': {}, 'files': {}}
    for group in REQUIREMENTS:
        wheels = sorted((bundle / 'wheels' / group).glob('*.whl'))
        if not wheels:
            raise ValueError(f'Empty wheelhouse: {group}')
        identities = set()
        entries = []
        for wheel in wheels:
            if not compatible_wheel(wheel.name):
                raise ValueError(f'Incompatible or untagged Linux wheel: {wheel.name}')
            name, version = wheel_identity(wheel)
            key = re.sub(r'[-_.]+', '-', name).lower()
            if key in identities:
                raise ValueError(f'Multiple candidate wheels for one package: {name}')
            identities.add(key)
            sha = digest(wheel)
            entry = {'filename': wheel.name, 'package': name, 'version': version, 'sha256': sha,
                     'binary_audit': audit_elf(wheel)}
            entries.append(entry)
        lock = '\n'.join(f'{entry["package"]}=={entry["version"]} --hash=sha256:{entry["sha256"]}' for entry in entries) + '\n'
        (bundle / f'{group}.lock.txt').write_text(lock, encoding='utf-8')
        document['groups'][group] = entries
    for file in sorted(bundle.rglob('*')):
        if file.is_file() and file.name != 'bundle.json':
            relative = file.relative_to(bundle).as_posix()
            contained(bundle, relative)
            document['files'][relative] = {'sha256': digest(file), 'bytes': file.stat().st_size}
    (bundle / 'bundle.json').write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    return document


def verify(bundle: Path) -> dict:
    document = json.loads((bundle / 'bundle.json').read_text(encoding='utf-8'))
    if document.get('schema') != 'ngv_wsl_bundle_v1' or document.get('python_sha256') != PYTHON_SHA or document.get('python') != '3.12.15':
        raise ValueError('Unknown bundle format/Python. Run bash setup_laptop.sh to rebuild the package cache.')
    if document.get('architecture') != 'x86_64' or document.get('glibc_min') != '2.27':
        raise ValueError('Bundle ABI baseline does not match this code release')
    expected_req = {path: digest(ROOT / path) for path in REQUIREMENTS.values()}
    if document.get('requirements') != expected_req:
        raise ValueError('Requirements changed after bundle preparation. Run bash setup_laptop.sh to refresh the package cache.')
    files = document.get('files', {})
    mandatory = {f'python/{PYTHON_ARCHIVE}', 'wsl_tf.lock.txt', 'wsl_onnx.lock.txt'}
    if not mandatory.issubset(files):
        raise ValueError('Incomplete bundle file inventory')
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file() and p.name != 'bundle.json'}
    if actual != set(files):
        raise ValueError(f'Bundle files changed: missing={sorted(set(files)-actual)}, extra={sorted(actual-set(files))}')
    for relative, info in files.items():
        path = contained(bundle, relative)
        if not SHA_RE.fullmatch(str(info.get('sha256', ''))) or not path.is_file() or path.stat().st_size != info['bytes'] or digest(path) != info['sha256']:
            raise ValueError(f'Bundle integrity failed: {relative}')
    if digest(bundle / 'python' / PYTHON_ARCHIVE) != PYTHON_SHA:
        raise ValueError('Official Python archive hash mismatch')
    for group in REQUIREMENTS:
        entries = document.get('groups', {}).get(group, [])
        paths = {f'wheels/{group}/{item["filename"]}' for item in entries}
        if not entries or paths != {name for name in files if name.startswith(f'wheels/{group}/')}:
            raise ValueError(f'Wheel inventory mismatch: {group}')
        for item in entries:
            if not compatible_wheel(item['filename']) or files[f'wheels/{group}/{item["filename"]}']['sha256'] != item['sha256']:
                raise ValueError(f'Invalid wheel metadata: {group}/{item["filename"]}')
        expected_lock = '\n'.join(f'{item["package"]}=={item["version"]} --hash=sha256:{item["sha256"]}' for item in entries) + '\n'
        if (bundle / f'{group}.lock.txt').read_text(encoding='utf-8') != expected_lock:
            raise ValueError(f'Install lock does not match wheel manifest: {group}')
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify'])
    parser.add_argument('--bundle', type=Path, default=ROOT / '.cache/wsl_packages')
    args = parser.parse_args()
    try:
        document = (build if args.action == 'build' else verify)(args.bundle)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        raise SystemExit(f'STOP: {error}') from error
    print('PACKAGE CACHE PASS: CPython', document['python'], ', glibc >=', document['glibc_min'])
    for group, entries in document['groups'].items():
        print(f'  {group}: {len(entries)} SHA256-locked compatible wheels')

if __name__ == '__main__':
    main()
