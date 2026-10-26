"""Train PINE and evaluate the checkpoint selected on validation MRR."""

import argparse
import copy
import json
import os
from pathlib import Path
import pprint
import sys

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from pine.data.protocol import read_splits, validate_splits
from pine.cache_validation import validate_processed

import torch
from torch_geometric.data import Data
from pine import util
from pine.models import PINE
from script import pretrain


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=f'{PROJECT_ROOT}/config/pine.yaml')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--setting', choices=['interpolation', 'extrapolation'], required=True)
    parser.add_argument('--time-range', type=int, required=True)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--num-epoch', type=int, default=10)
    parser.add_argument('--processed-root', help='Override the processed graph root')
    parser.add_argument('--query-time', choices=['on', 'off'], default='on')
    args = parser.parse_args()
    if args.num_epoch < 1 or args.time_range < 1:
        parser.error('--num-epoch and --time-range must be positive')
    return args


def main():
    args = parse_args()
    context = {'project_root': PROJECT_ROOT, 'dataset': args.dataset,
               'setting': args.setting, 'time_range': args.time_range}
    cfg = util.load_config(args.config, context=context)
    cfg.output_dir = os.path.join(cfg.output_dir, args.query_time)
    cfg.train.num_epoch = args.num_epoch
    if args.processed_root:
        cfg.dataset.processed_root = os.path.abspath(os.path.expanduser(args.processed_root))
    raw_dir = os.path.join(cfg.dataset.raw_root, cfg.type, cfg.d)
    processed_dir = os.path.join(cfg.dataset.processed_root, cfg.type, cfg.d)
    validate_splits(read_splits(raw_dir))
    marker = validate_processed(raw_dir, processed_dir, cfg.type)
    if marker.get('storage', {}).get('msg') == 'full_partitioned':
        from pine.full_partitions import install_full_message_runtime
        install_full_message_runtime(pretrain)
    working_dir = util.create_working_directory(cfg)
    torch.manual_seed(args.seed)
    pretrain.logger = util.get_root_logger()
    logger = pretrain.logger
    logger.warning('Config:\n%s', pprint.pformat(cfg))

    train_data, all_train = pretrain.load_file(os.path.join(raw_dir, 'train.txt'))
    valid_data, all_valid = pretrain.load_file(os.path.join(raw_dir, 'valid.txt'))
    test_data, all_test = pretrain.load_file(os.path.join(raw_dir, 'test.txt'))
    _, all_msg = pretrain.load_file(os.path.join(raw_dir, 'msg.txt'))
    graph = pretrain.load_graph_metadata(processed_dir, 'msg')
    facts = torch.tensor(all_train + all_valid + all_test + all_msg, dtype=torch.long)
    edges, relations, times = facts[:, :2].t(), facts[:, 2], facts[:, 3]
    filtered_data = Data(
        edge_index=torch.cat([edges, edges.flip(0)], dim=1),
        edge_type=torch.cat([relations, relations + graph.num_relations // 2]),
        time=torch.cat([times, times]), num_nodes=graph.num_nodes,
    )
    model = PINE(copy.deepcopy(cfg.model.relation_model),
                 copy.deepcopy(cfg.model.entity_model),
                 query_time_range=args.time_range,
                 use_query_time=args.query_time == 'on').cuda()
    best_epoch, best_mrr = pretrain.train_and_validate(
        cfg, model, train_data, valid_data, processed_dir,
        batch_size=cfg.train.batch_size, filtered_data=filtered_data,
        batch_per_epoch=cfg.train.batch_per_epoch,
    )
    metrics = pretrain.evaluate(model, test_data, processed_dir,
                               batch_size=cfg.train.batch_size, filtered_data=filtered_data)
    result = {'dataset': args.dataset, 'setting': args.setting, 'seed': args.seed,
              'num_epoch': args.num_epoch, 'query_time_enabled': args.query_time == 'on',
              'protocol': marker['protocol'], 'source_sha256': marker['source_sha256'],
              'best_epoch': best_epoch, 'best_validation_mrr': best_mrr, 'test': metrics}
    output = Path(working_dir) / 'metrics.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    logger.warning('Saved evaluation results to %s', output)


if __name__ == '__main__':
    main()
