"""Losslessly package original Phase 2 trial JSON into a new gzip archive.

Original files are never edited/deleted. Each compressed record retains an explicit
checksum of its original bytes. verify_completed.py reads either representation.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def archive(out):
    source=ROOT/'results/controlled_care_v1'
    out.mkdir(parents=True,exist_ok=False)
    receipts=[]
    for objective in ('survival','offspring'):
        for seed in range(10):
            name=f'{objective}_s{seed}.json'
            original=(source/name).read_bytes()
            compressed=gzip.compress(original,compresslevel=9,mtime=0)
            assert gzip.decompress(compressed)==original
            (out/f'{name}.gz').write_bytes(compressed)
            receipts.append({'source':f'results/controlled_care_v1/{name}',
                             'archive':f'{name}.gz','source_bytes':len(original),
                             'archive_bytes':len(compressed),
                             'source_sha256':hashlib.sha256(original).hexdigest(),
                             'archive_sha256':hashlib.sha256(compressed).hexdigest()})
    manifest={'format':'gzip of byte-identical original JSON; mtime=0',
              'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
              'source_summary_sha256':hashlib.sha256((source/'summary.json').read_bytes()).hexdigest(),
              'files':receipts,'source_bytes':sum(r['source_bytes'] for r in receipts),
              'archive_bytes':sum(r['archive_bytes'] for r in receipts)}
    (out/'ARCHIVE.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/controlled_care_v1_archive')
    archive(parser.parse_args().out)
