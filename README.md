# PINE

PINE is a framework for temporal knowledge graph completion with relation-level
and entity-level message passing. Relation messages encode event-to-event
intervals, while entity messages incorporate the signed interval between each
support fact and the query timestamp.

## Installation

Use Python 3.10 and a CUDA-capable GPU. Install the PyTorch and `torch-scatter`
wheels matching your CUDA environment before installing the remaining packages.
For CUDA 12.1:

```bash
conda create -n pine python=3.10 -y
conda activate pine
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install -r requirements.txt
```

## Datasets

The datasets are provided under `dataset/<setting>/<name>/`.

| Setting | Dataset | Time range |
| --- | --- | ---: |
| Interpolation | ICEWS14 | 365 |
| Interpolation | ICEWS05-15 | 4017 |
| Interpolation | GDELT | 366 |
| Extrapolation | ACLED | 336 |
| Extrapolation | icews21 | 365 |
| Extrapolation | icews22 | 365 |

Each dataset includes `train.txt`, `msg.txt`, `valid.txt`, `test.txt`, and ID
mappings. Facts use tab-separated `head relation tail timestamp` IDs. The `msg`
split provides evaluation support; validation and test targets are excluded
from training and support. Entity and relation vocabularies may overlap across
splits. Extrapolation uses only support facts strictly preceding the query.

## Training and Evaluation

Run commands from the repository root. For ICEWS14:

```bash
python -m pine.preprocess \
  --dataset ICEWS14 --setting interpolation \
  --data-dir dataset/interpolation/ICEWS14 \
  --output-dir pine_data/processed/interpolation/ICEWS14 \
  --shard-msg

CUDA_VISIBLE_DEVICES=0 python script/train.py \
  --dataset ICEWS14 --setting interpolation --time-range 365 \
  --num-epoch 10 --seed 1024
```

Hyperparameters are defined in `config/pine.yaml`. Preprocessing defaults to
30 query timestamps per shard, 60 context timestamps, and at most 5,000 support
facts. For other datasets, use the setting and time range listed above.

To preprocess and train all six datasets sequentially on one GPU:

```bash
GPU_IDS="0" MAX_PARALLEL=1 bash script/run_all.sh full
```

For six GPUs, run `bash script/run_all.sh full`; each dataset uses one GPU.
Set `NUM_EPOCHS` and `SEED` to override the defaults of 10 and 1024.

The checkpoint with the highest validation MRR is evaluated on the test split.
Outputs are saved under `pine_runs/`, including `model_best.pth` and
`metrics.json`. Evaluation reports filtered MR, MRR, and Hits@1/3/10 for head
and tail prediction. The batch launcher also produces `six_dataset_results.csv`.

## Validation

```bash
python script/validate_data.py
pip install -r requirements-dev.txt
python -m pytest -q
```

Native model tests require `torch-geometric` and `torch-scatter`. CPU adapter
tests also cover temporal scoring and gradients without those extensions.
