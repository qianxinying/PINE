import sys
import os
import json
import torch
from torch import optim
from torch import nn
from torch import distributed as dist

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from pine import tasks, util
from pine.data.protocol import assert_causal_support, read_splits, validate_splits


separator = ">" * 30
line = "-" * 30


class GraphStore:
    def __init__(self, processed_dir, split, device="cuda:0"):
        self.processed_dir = processed_dir
        self.split = split
        self.device = device
        self.current_file = None
        self.current_graph = None
        self.current_support_max = None
        self.setting = None
        manifest_path = os.path.join(processed_dir, f"{split}_manifest.json")
        storage_marker = os.path.join(processed_dir, "storage_format.json")
        self.time_to_file = None
        storage = None
        if os.path.isfile(storage_marker):
            with open(storage_marker, encoding="utf-8") as fin:
                marker = json.load(fin)
                storage = marker.get("storage")
                self.setting = marker.get("setting")
        if isinstance(storage, dict):
            split_storage = storage.get(self.split)
        else:
            # Backward compatibility with the original dataset-level marker.
            split_storage = storage
        if split_storage == "sharded" and os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as fin:
                manifest = json.load(fin)
                self.time_to_file = manifest["time_to_file"]
                self.setting = self.setting or manifest.get("setting")
        elif split_storage is None and os.path.isfile(manifest_path):
            # Backward compatibility for directories created before the marker.
            with open(manifest_path, encoding="utf-8") as fin:
                manifest = json.load(fin)
                self.time_to_file = manifest["time_to_file"]
                self.setting = self.setting or manifest.get("setting")
        if self.setting is None:
            parent_setting = os.path.basename(os.path.dirname(os.path.abspath(processed_dir)))
            if parent_setting in ("interpolation", "extrapolation"):
                self.setting = parent_setting

    def graph_path(self, timestamp):
        if self.time_to_file is None:
            return os.path.join(self.processed_dir, f"{self.split}.pt")
        key = str(int(timestamp))
        if key not in self.time_to_file:
            raise KeyError(
                f"No {self.split} graph shard for timestamp {timestamp}"
            )
        return os.path.join(self.processed_dir, self.time_to_file[key])

    def get(self, timestamp):
        graph_file = self.graph_path(timestamp)
        if graph_file != self.current_file:
            self.clear()
            # These files contain PyG Data objects created by our preprocessing.
            graph = torch.load(graph_file, map_location="cpu", weights_only=False)
            self.current_support_max = int(graph.time.max()) if graph.time.numel() else None
            if self.split == "msg" and self.setting == "extrapolation":
                assert_causal_support(self.current_support_max, timestamp)
            self.current_graph = graph.to(self.device)
            self.current_file = graph_file
            logger.info("Loaded %s graph shard: %s", self.split, graph_file)
        if self.split == "msg" and self.setting == "extrapolation":
            assert_causal_support(self.current_support_max, timestamp)
        return self.current_graph

    def clear(self):
        if self.current_graph is not None:
            del self.current_graph
            self.current_graph = None
            self.current_file = None
            self.current_support_max = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def load_graph_metadata(processed_dir, split="msg"):
    store = GraphStore(processed_dir, split, device="cpu")
    if store.time_to_file:
        first_time = min(store.time_to_file, key=int)
    else:
        first_time = 0
    return torch.load(store.graph_path(first_time), map_location="cpu", weights_only=False)

def train_and_validate(cfg, model, train_data, valid_data, processed_dir, batch_size=16,
                       filtered_data=None, batch_per_epoch=None):
    device='cuda:0'
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    best_result = float("-inf")
    best_epoch = -1
    batch_id = 0
    graph_store = GraphStore(processed_dir, "train", device=device)
    for epoch in range(0, cfg.train.num_epoch):
        parallel_model.train()
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Epoch %d begin" % epoch)
        epoch_loss = []
        for time_range in range(len(train_data)):
            if time_range > 0:
                del train_graph
            query_time = train_data[time_range][0][3]
            train_graph = graph_store.get(query_time)
            losses=[]
            for j in range(0, len(train_data[time_range]), batch_size):
                batch = train_data[time_range][j : min(j +batch_size, len(train_data[time_range]))]
                if len(batch)==0:
                    continue
                batch = torch.tensor(batch)
                batch = batch.unsqueeze(1).cuda()
                batch = tasks.negative_sampling(train_graph, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_graph, batch)
                loss = parallel_model.compute_loss(pred)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
                batch_id += 1
            avg_loss = sum(losses) / len(losses)
            epoch_loss.append(avg_loss)
            logger.warning("binary cross entropy: %g" % avg_loss)

        if util.get_rank() == 0:
            avg_loss = sum(losses) / len(losses)
            logger.warning(separator)
            logger.warning("Epoch %d end" % epoch)
            logger.warning(line)
            epoch_loss_num = sum(epoch_loss)/len(epoch_loss)
            logger.warning("average binary cross entropy: %g" % epoch_loss_num)

        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        del train_graph
        graph_store.clear()
        metrics = evaluate(model, valid_data, processed_dir, batch_size=batch_size,
                           filtered_data=filtered_data)
        result = metrics["mrr"]
        print(metrics)
        if result > best_result:
            best_result = result
            best_epoch = epoch
            torch.save(state, "model_best.pth")

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_best.pth", map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()
    return best_epoch, best_result

@torch.no_grad()
def evaluate(model, test_data, processed_dir, batch_size=16, filtered_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()
    device='cuda:0'
    rankings = []
    num_negatives = []
    graph_store = GraphStore(processed_dir, "msg", device=device)

    for time_range in range(len(test_data)):
        if time_range > 0:
            del test_graph
        query_time = test_data[time_range][0][3]
        test_graph = graph_store.get(query_time)
        model.eval()
        for j in range(0, len(test_data[time_range]), batch_size):
            batch_list = test_data[time_range][j : min(j + batch_size, len(test_data[time_range]))]
            if len(batch_list)==0:
                continue

            batch = torch.tensor(batch_list).cuda()
            t_batch, h_batch = tasks.all_negative(test_graph, batch)
            t_pred= model(test_graph, t_batch)
            h_pred= model(test_graph, h_batch)
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
            pos_h_index, pos_t_index, pos_r_index,time = batch.t()
            t_ranking,t_pred_result = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking, h_pred_result = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            num_t_negative = t_mask.sum(dim=-1)
            num_h_negative = h_mask.sum(dim=-1)
            rankings += [t_ranking, h_ranking]
            num_negatives += [num_t_negative, num_h_negative]
    del test_graph
    graph_store.clear()
    ranking = torch.cat(rankings)
    print(ranking.shape)
    num_negative = torch.cat(num_negatives)
    print(num_negative.shape)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

    metrics = {
        "mr": all_ranking.float().mean().item(),
        "mrr": (1 / all_ranking.float()).mean().item(),
        "hits@1": (all_ranking <= 1).float().mean().item(),
        "hits@3": (all_ranking <= 3).float().mean().item(),
        "hits@10": (all_ranking <= 10).float().mean().item(),
    }
    if rank == 0:
        for metric, score in metrics.items():
            logger.warning("%s: %g" % (metric, score))
    if rank == 0:
        logger.warning(separator)
    return metrics


def load_file(triplet_file):
    if os.path.basename(triplet_file) == "train.txt":
        validate_splits(read_splits(os.path.dirname(os.path.abspath(triplet_file))))
    all_triplets=[]
    triplets = {}

    with open(triplet_file, "r", encoding="utf-8") as fin:
        for l in fin:
            u, r,v,t = l.strip().split('\t')
            all_triplets.append([int(u), int(v), int(r),int(t)])
            if int(t) in triplets.keys():
                triplets[int(t)].append([int(u), int(v), int(r),int(t)])
            else:
                triplets[int(t)] = [[int(u), int(v), int(r),int(t)]]
    triplets = sorted(triplets.items(), key=lambda x: x[0])
    triplets = [group for _, group in triplets]
    return triplets,all_triplets
