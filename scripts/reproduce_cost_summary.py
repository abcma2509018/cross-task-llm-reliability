#!/usr/bin/env python3
"""REPRODUCTION UTILITY, NOT ORIGINAL MEASUREMENT SCRIPT.
Summarizes existing *_progress.json files without rerunning extraction.
"""
import json, pathlib
for p in pathlib.Path('.').rglob('*_progress.json'):
    try:
        x=json.loads(p.read_text())
    except Exception:
        continue
    if x.get('processed') and x.get('elapsed_seconds'):
        print(f'{p}: {1000*x["elapsed_seconds"]/x["processed"]:.3f} ms/sample')
