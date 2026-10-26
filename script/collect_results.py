"""Collect PINE evaluation metrics into a CSV table."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = []
    for path in Path(args.run_root).expanduser().resolve().rglob('metrics.json'):
        result = json.loads(path.read_text(encoding='utf-8'))
        row = {key: result[key] for key in ('setting', 'dataset', 'seed', 'num_epoch',
               'query_time_enabled', 'best_epoch', 'best_validation_mrr')}
        row.update({f'test_{key}': value for key, value in result['test'].items()})
        rows.append(row)
    rows.sort(key=lambda row: (row['setting'], row['dataset'], row['seed']))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ['setting', 'dataset', 'seed']
    with output.open('w', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Collected {len(rows)} runs into {output}')


if __name__ == '__main__':
    main()
