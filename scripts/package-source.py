#!/usr/bin/env python3
"""Archive the exact plugin and pinned JUCE sources alongside binary artifacts."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
JUCE_COMMIT = 'c352e2489000ffeb9e397051377d75eef2efe641'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--juce-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new archive path')
    if git(args.juce_source, 'rev-parse', 'HEAD').decode().strip() != JUCE_COMMIT:
        parser.error('JUCE checkout is not the pinned 8.0.11 source')
    receipt = {'plugin_commit': git(ROOT, 'rev-parse', 'HEAD').decode().strip(),
               'juce_commit': JUCE_COMMIT, 'sha256': {}}
    for root in (ROOT, args.juce_source):
        if git(root, 'status', '--porcelain', '--untracked-files=no').strip():
            parser.error('Archive committed source only: ' + str(root))
    with tarfile.open(args.output, 'x:gz') as archive:
        for prefix, root in [('stemgenrt', ROOT), ('JUCE', args.juce_source)]:
            for raw in git(root, 'ls-files', '-z').split(b'\0'):
                if not raw:
                    continue
                name = raw.decode()
                source = root / name
                if not source.is_file():
                    raise ValueError('Missing source file: ' + name)
                data = source.read_bytes()
                if data.startswith(b'version https://git-lfs.github.com/spec/'):
                    raise ValueError('Fetch LFS payload before archiving: ' + name)
                key = prefix + '/' + name
                receipt['sha256'][key] = hashlib.sha256(data).hexdigest()
                archive.add(source, arcname=key, recursive=False)
        data = (json.dumps(receipt, indent=2) + '\n').encode()
        info = tarfile.TarInfo('source-identity.json'); info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    print(json.dumps({'archive': args.output.name,
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'files': len(receipt['sha256'])}))


if __name__ == '__main__':
    main()
