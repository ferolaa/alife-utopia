"""Lossless Phase 3 packaging/restoration; refuses to overwrite destination paths.

Archive and index are inseparable. All members, sizes and SHA-256 hashes are checked
before restoration; only flat JSON names are accepted. No simulation is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def verify(archive, index):
    info = json.loads(Path(index).read_text())
    assert digest(Path(archive).read_bytes()) == info['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) and set(names) == set(info['records'])
        for name in names:
            assert Path(name).name == name and name.endswith('.json')
            payload = z.read(name)
            assert len(payload) == info['records'][name]['bytes']
            assert digest(payload) == info['records'][name]['sha256']
    return info


def pack(source, destination):
    source, destination = Path(source), Path(destination)
    assert source.is_dir() and not destination.exists()
    files = sorted(source.iterdir())
    assert files and all(p.is_file() and p.suffix == '.json' for p in files)
    destination.mkdir(parents=True)
    archive = destination / 'records.zip'
    records = {}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in files:
            payload = path.read_bytes()
            member = zipfile.ZipInfo(path.name, (1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            z.writestr(member, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            records[path.name] = {'bytes': len(payload), 'sha256': digest(payload)}
    info = {'format': 'ZIP DEFLATE, lossless original bytes', 'source_directory': source.name,
            'files': len(records), 'trajectories': len(records)-2,
            'original_bytes': sum(r['bytes'] for r in records.values()),
            'archive_bytes': archive.stat().st_size, 'archive_sha256': digest(archive.read_bytes()),
            'records': records}
    index = destination / 'ARCHIVE.json'
    index.write_text(json.dumps(info, indent=2)+'\n')
    verify(archive, index)
    return info


def restore(archive, index, destination):
    info = verify(archive, index)
    destination = Path(destination)
    assert not destination.exists(), 'Restoration requires a NEW directory'
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for name in info['records']:
            path = destination / name
            with path.open('xb') as f:
                f.write(z.read(name))
            assert digest(path.read_bytes()) == info['records'][name]['sha256']
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pack'); p.add_argument('source', type=Path); p.add_argument('destination', type=Path)
    for command in ('verify', 'restore'):
        p = sub.add_parser(command); p.add_argument('archive', type=Path); p.add_argument('index', type=Path)
        if command == 'restore': p.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.command == 'pack': info = pack(args.source, args.destination)
    elif args.command == 'restore': info = restore(args.archive, args.index, args.destination)
    else: info = verify(args.archive, args.index)
    print(json.dumps({k:v for k,v in info.items() if k != 'records'}, indent=2))
