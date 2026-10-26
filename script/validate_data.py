"""Validate numeric splits and ID mappings without training dependencies."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pine.data.protocol import read_splits, validate_splits


def read_ids(path):
    identifiers = []
    labels = set()
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        label, identifier = line.rsplit('\t', 1)
        if not label or label in labels:
            raise ValueError(f'{path}:{number}: empty or duplicate label')
        labels.add(label)
        identifiers.append(int(identifier))
    if sorted(identifiers) != list(range(len(identifiers))):
        raise ValueError(f'{path}: expected unique contiguous IDs starting at zero')
    return set(identifiers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'dataset')
    args = parser.parse_args()
    directories = sorted(args.data_root.glob('*/*/train.txt'))
    if not directories:
        raise FileNotFoundError(f'No datasets found in {args.data_root}')
    for path in directories:
        directory = path.parent
        splits = read_splits(directory)
        validate_splits(splits)
        entity_ids = read_ids(directory / 'entity2id.txt')
        relation_path = directory / 'rel2id.txt'
        if not relation_path.exists():
            relation_path = directory / 'relation2id.txt'
        relation_ids = read_ids(relation_path)
        time_ids = read_ids(directory / 'time2id.txt')
        for split, rows in splits.items():
            for h, r, t, timestamp in rows:
                if h not in entity_ids or t not in entity_ids or r not in relation_ids or timestamp not in time_ids:
                    raise ValueError(f'{directory}/{split}.txt: unmapped ID')
        counts = ', '.join(f'{split}={len(rows)}' for split, rows in splits.items())
        print(f'[PASS] {directory.relative_to(args.data_root)}: {counts}')
    print(f'All {len(directories)} datasets passed.')


if __name__ == '__main__':
    main()
