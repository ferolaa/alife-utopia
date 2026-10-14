"""Lossless frozen-neural evaluation archive; no simulation or learning.

Uses the committed archive verifier/restorer for flat JSON records and hashes.
Pilot records are an exact subset of the full records and can be reconstructed
byte-identically, avoiding a second copy of those trajectories in Git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.archive_developmental_care import verify, restore


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def pack(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists(): raise FileExistsError('Archive target must be NEW')
    names = {'manifest.json','records.json','summary.json','INTEGRITY.json'}
    assert {p.name for p in source.iterdir()} == names
    manifest = json.loads((source/'manifest.json').read_text())
    records = json.loads((source/'records.json').read_text())
    assert manifest['expected_trials'] == len(records) == 2000
    assert len({(r['training_seed'],r['world_seed'],r['energy'],r['cell']) for r in records}) == 2000
    for name,h in json.loads((source/'INTEGRITY.json').read_text()).items():
        assert sha((source/name).read_bytes()) == h
    destination.mkdir(parents=True)
    archive = destination/'records.zip'
    members = {}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted(names):
            payload = (source/name).read_bytes()
            item = zipfile.ZipInfo(name,(1980,1,1,0,0,0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            z.writestr(item,payload,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
            members[name] = {'bytes':len(payload),'sha256':sha(payload)}
    info = {'format':'ZIP DEFLATE, lossless original bytes','source_directory':source.name,
            'files':4,'trials':2000,'original_bytes':sum(p['bytes'] for p in members.values()),
            'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive.read_bytes()),'records':members}
    (destination/'ARCHIVE.json').write_text(json.dumps(info,indent=2)+'\n')
    verify(archive,destination/'ARCHIVE.json')
    return info


def restore_pilot(full_data, pilot_metadata, destination):
    full_data,pilot_metadata,destination=map(Path,(full_data,pilot_metadata,destination))
    if destination.exists(): raise FileExistsError('Pilot restoration target must be NEW')
    m=json.loads((pilot_metadata/'manifest.json').read_text())
    records=json.loads((full_data/'records.json').read_text())
    subset=[r for r in records if r['training_seed'] in m['training_seeds'] and r['world_seed'] in m['world_seeds']]
    assert len(subset)==m['expected_trials']==60
    payload=(json.dumps(subset,indent=2)+'\n').encode()
    integrity=json.loads((pilot_metadata/'INTEGRITY.json').read_text())
    assert sha(payload)==integrity['records.json']
    for name in ['manifest.json','summary.json']:assert sha((pilot_metadata/name).read_bytes())==integrity[name]
    destination.mkdir(parents=True)
    for name in ['manifest.json','summary.json','INTEGRITY.json']:shutil.copyfile(pilot_metadata/name,destination/name)
    (destination/'records.json').write_bytes(payload)
    return {'restored_pilot_trials':60,'records_sha256':sha(payload)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('pack');q.add_argument('source',type=Path);q.add_argument('destination',type=Path)
    for command in ['verify','restore']:
        q=sub.add_parser(command);q.add_argument('archive',type=Path);q.add_argument('index',type=Path)
        if command=='restore':q.add_argument('destination',type=Path)
    q=sub.add_parser('restore-pilot');q.add_argument('full_data',type=Path);q.add_argument('pilot_metadata',type=Path);q.add_argument('destination',type=Path)
    a=p.parse_args()
    if a.command=='pack': result=pack(a.source,a.destination)
    elif a.command=='restore':result=restore(a.archive,a.index,a.destination)
    elif a.command=='verify':result=verify(a.archive,a.index)
    else:result=restore_pilot(a.full_data,a.pilot_metadata,a.destination)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
