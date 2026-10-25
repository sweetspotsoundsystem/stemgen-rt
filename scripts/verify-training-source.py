#!/usr/bin/env python3
"""Check the bundled complete training source against the plugin version."""
import hashlib
import json
from pathlib import Path
import re
import tarfile


def verify(root):
    root = Path(root)
    version = re.search(r'project\(AudioPlugin VERSION (\d+\.\d+\.\d+)\)',
                        (root / 'plugin/CMakeLists.txt').read_text()).group(1)
    folder = root / 'model/training-source'
    manifest = json.loads((folder / 'sha256.json').read_text())
    filename = f'stemgenrt-{version}.tar.gz'
    if manifest.get('release') != 'v' + version:
        raise ValueError('Training source version differs from the plugin')
    for key in ('public_source_commit', 'public_source_tree'):
        if not re.fullmatch(r'[0-9a-f]{40}', manifest.get(key, '')):
            raise ValueError('Training source needs an exact public commit and tree')
    if manifest.get('public_source_repository') != 'https://github.com/sweetspotsoundsystem/StemgenRT-5.8':
        raise ValueError('Training source repository differs')
    archive = folder / filename
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest.get(filename):
        raise ValueError('Training source archive checksum differs')
    prefix = f'stemgenrt-{version}/'
    with tarfile.open(archive) as stream:
        files = {}
        for member in stream.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or not member.name.startswith(prefix):
                raise ValueError('Training source archive has an unexpected member')
            name = member.name[len(prefix):]
            if (Path(name).is_absolute() or '..' in Path(name).parts or name in files
                    or {'paper', 'papers', 'research', 'runs', 'datasets'}.intersection(Path(name).parts)
                    or name.endswith(('.pt', '.onnx'))):
                raise ValueError('Training source archive contains an excluded or duplicate member')
            files[name] = stream.extractfile(member).read()
    required = {
        'stemgenrt/' + name + '.py' for name in (
            'independent', 'banded', 'specialist', 'trainer', 'losses', 'data',
            'checkpoint', 'recovery', 'evaluation', 'independent_export', 'partition_independent')}
    required.update({'configs/vocals-independent.json', 'configs/bass-specialist.json',
                     'configs/drums-specialist.json', 'scripts/export_independent_model.py',
                     'train_streaming.py', 'docs/training.md', 'docs/independent-stems.md',
                     'tests/test_partition_independent.py', 'LICENSE', 'PKG-INFO'})
    if not required.issubset(files):
        raise ValueError('Matching model/training/export support is incomplete: '
                         + ', '.join(sorted(required - files.keys())))
    if f'Version: {version}' not in files['PKG-INFO'].decode().splitlines():
        raise ValueError('Packaged Python version differs from the plugin')
    return {'version': version, 'archive': filename, 'files': len(files),
            'public_source_commit': manifest['public_source_commit']}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
