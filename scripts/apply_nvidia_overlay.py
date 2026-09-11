#!/usr/bin/env python3
"""Apply only the hash-pinned NVIDIA compatibility delta to the accepted base."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def apply(root, patch, manifest):
 entries=json.loads(manifest.read_text())
 for name,pins in entries.items():
  path=root/name
  if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe manifest path')
  if pins['before'] is None:
   if path.exists():raise ValueError(f'new file already exists: {name}')
  elif not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=pins['before']:
   raise ValueError(f'base source mismatch: {name}')
 subprocess.run(['git','apply','--check',str(patch.resolve())],cwd=root,check=True)
 subprocess.run(['git','apply',str(patch.resolve())],cwd=root,check=True)
 for name,pins in entries.items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=pins['after']:raise ValueError(f'patched source mismatch: {name}')
 print(f'Verified {len(entries)} patched source/test files.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--patch',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();apply(a.root,a.patch,a.manifest)
