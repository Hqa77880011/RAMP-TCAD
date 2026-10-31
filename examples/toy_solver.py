"""Analytic schema fixture only: this is not a TCAD device simulator."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
parser.add_argument('--response', required=True)
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
x = request['x'][0]
bias = 0.03 if request['fidelity'] != 'fine' else 0.
Path(args.response).write_text(json.dumps(dict(completed=True, objectives=[x + bias, 1 - x], constraints=[-1.])))
