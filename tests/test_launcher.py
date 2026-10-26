"""Run the actual shell launchers with simulated preprocessing/training."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from pine.data.protocol import PROTOCOL, source_fingerprints


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = self.root / "script"
        scripts.mkdir()
        source = Path(__file__).resolve().parents[1] / "script"
        for name in ("run_all.sh", "run_all_nohup.sh"):
            shutil.copyfile(source / name, scripts / name)
        (self.root / "pine/data").mkdir(parents=True)
        for name in ("__init__.py", "cache_validation.py"):
            shutil.copyfile(source.parent / "pine" / name, self.root / "pine" / name)
        for name in ("__init__.py", "protocol.py"):
            shutil.copyfile(source.parent / "pine/data" / name, self.root / "pine/data" / name)
        self.datasets = ["icews21", "icews22", "ACLED", "ICEWS14", "ICEWS05-15", "GDELT"]
        for index, dataset in enumerate(self.datasets):
            setting = "extrapolation" if index < 3 else "interpolation"
            raw = self.root / "dataset" / setting / dataset
            raw.mkdir(parents=True)
            for timestamp, split in enumerate(("train", "msg", "valid", "test")):
                (raw / f"{split}.txt").write_text(f"0\t0\t1\t{timestamp}\n")
        (self.root / "calls").mkdir()
        fake_python = self.root / "fake_python"
        fake_python.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-':
    sys.argv = args
    exec(compile(sys.stdin.read(), '<launcher verification>', 'exec'))
    sys.exit(0)
kind = args[1].split('.')[-1] if args[0] == '-m' else Path(args[0]).stem
dataset = args[args.index('--dataset') + 1] if '--dataset' in args else 'all'
record = {'args': args, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
          'cwd': os.getcwd()}
path = Path(os.environ['PROJECT_ROOT']) / 'calls' / (kind + '_' + dataset + '.json')
path.write_text(json.dumps(record))
if kind == 'preprocess':
    output = Path(args[args.index('--output-dir') + 1])
    marker_path = output / 'storage_format.json'
    if os.environ.get('LEGACY_PREPROCESS'):
        marker_path.write_text(json.dumps({'format': 'pine_storage_v1',
                                          'dataset': dataset, 'storage': {'train': 'sharded', 'msg': 'monolithic'}}))
if kind == 'train' and dataset == os.environ.get('FAIL_DATASET'):
    sys.exit(1)
''')
        fake_python.chmod(0o755)
        self.processed = self.root / "fresh graphs"
        for index, dataset in enumerate(self.datasets):
            setting = "extrapolation" if index < 3 else "interpolation"
            processed = self.processed / setting / dataset
            processed.mkdir(parents=True)
            raw = self.root / "dataset" / setting / dataset
            (processed / "storage_format.json").write_text(json.dumps({
                "protocol": PROTOCOL, "setting": setting,
                "storage": {"train": "sharded", "msg": "sharded"},
                "source_sha256": source_fingerprints(raw),
            }))
            for split in ("train", "msg"):
                (processed / f"{split}_manifest.json").write_text(json.dumps({
                    "protocol": PROTOCOL, "setting": setting, "causal_evaluation": True,
                    "shards": [{"support_time_max": 1, "query_time_min": 2}],
                }))
        self.env = {**os.environ, "PROJECT_ROOT": str(self.root),
                    "PYTHON": str(fake_python), "GPU_IDS": "2 3 4 5 6 7",
                    "NUM_EPOCHS": "3", "PROCESSED_ROOT": str(self.processed)}

    def run_launcher(self, name, action="full"):
        return subprocess.run(["bash", str(self.root / "script" / name), action],
                              env=self.env, text=True, capture_output=True, timeout=15)

    def calls(self):
        return {path.stem: json.loads(path.read_text()) for path in (self.root / "calls").glob("*.json")}

    def test_nohup_inherits_six_gpu_mapping_epochs_and_custom_graph_root(self):
        result = self.run_launcher("run_all_nohup.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        log = Path(next(line[5:] for line in result.stdout.splitlines() if line.startswith("Log: ")))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if log.exists() and "[results]" in log.read_text():
                break
            time.sleep(0.02)
        else:
            self.fail(f"Background launcher did not finish: {log.read_text()}")
        calls = self.calls()
        self.assertEqual(len(calls), 13)  # six preprocess, six train, one collect
        for index, dataset in enumerate(self.datasets):
            preprocess = calls[f"preprocess_{dataset}"]["args"]
            self.assertEqual(preprocess[preprocess.index("--output-dir") + 1],
                             str(self.processed / ("extrapolation" if index < 3 else "interpolation") / dataset))
            train = calls[f"train_{dataset}"]
            self.assertEqual(train["gpu"], str(index + 2))
            self.assertEqual(train["args"][train["args"].index("--num-epoch") + 1], "3")
            self.assertEqual(train["args"][train["args"].index("--processed-root") + 1], str(self.processed))
            self.assertTrue(train["cwd"].endswith(f"work_{dataset}"))
        text = log.read_text()
        self.assertLess(text.index("[preprocess done] GDELT"), text.index("[launch]"))

    def test_failed_training_is_reported_and_results_are_not_collected(self):
        self.env["FAIL_DATASET"] = "ACLED"
        result = self.run_launcher("run_all.sh", "experiment")
        self.assertEqual(result.returncode, 1)
        self.assertIn("[failed] ACLED", result.stderr)
        self.assertEqual(len(self.calls()), 6)
        self.assertNotIn("collect_results_all", self.calls())

    def test_insufficient_gpu_list_stops_before_processing(self):
        self.env["GPU_IDS"] = "2 3 4"
        result = self.run_launcher("run_all.sh")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Expected at least 6 GPU IDs", result.stderr)
        self.assertEqual(self.calls(), {})

    def test_old_preprocessor_is_caught_after_first_dataset_before_gpu_launch(self):
        self.env["LEGACY_PREPROCESS"] = "1"
        result = self.run_launcher("run_all.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("found protocol=None, setting=None", result.stderr)
        self.assertEqual(set(self.calls()), {"preprocess_icews21"})
        self.assertNotIn("[launch]", result.stdout)

    def test_experiment_checks_all_datasets_before_starting_any_gpu_process(self):
        marker = self.processed / "interpolation" / "GDELT" / "storage_format.json"
        marker.write_text(json.dumps({"format": "pine_storage_v1"}))
        result = self.run_launcher("run_all.sh", "experiment")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("found protocol=None, setting=None", result.stderr)
        self.assertEqual(self.calls(), {})

    def test_off_switch_is_forwarded_to_all_six_datasets_in_two_gpu_batches(self):
        self.env.update(QUERY_TIME="off", MAX_PARALLEL="2", GPU_IDS="4 7")
        result = self.run_launcher("run_all.sh", "experiment")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 7)
        for index, dataset in enumerate(self.datasets):
            call = calls[f"train_{dataset}"]
            self.assertEqual(call["gpu"], "4" if index % 2 == 0 else "7")
            self.assertEqual(call["args"][call["args"].index("--query-time") + 1], "off")


if __name__ == "__main__":
    unittest.main()
