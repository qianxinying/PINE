"""Check experiment startup without importing the GPU training stack."""

import ast
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

from pine.data.protocol import PROTOCOL, source_fingerprints


class ExperimentValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.raw = root / "raw" / "interpolation" / "fixture"
        self.processed = root / "processed" / "interpolation" / "fixture"
        self.raw.mkdir(parents=True)
        self.processed.mkdir(parents=True)
        for index, split in enumerate(("train", "msg", "valid", "test")):
            (self.raw / f"{split}.txt").write_text(f"0\t0\t1\t{index}\n")
        (self.processed / "storage_format.json").write_text(json.dumps({
            "protocol": PROTOCOL,
            "setting": "interpolation",
            "storage": {"train": "single", "msg": "single"},
            "source_sha256": source_fingerprints(self.raw),
        }))
        cfg = types.SimpleNamespace(
            output_dir=str(root / "runs"),
            train=types.SimpleNamespace(), type="interpolation", d="fixture",
            dataset=types.SimpleNamespace(raw_root=str(root / "raw"),
                                          processed_root=str(root / "processed")),
        )
        args = types.SimpleNamespace(dataset="fixture",
                                     setting="interpolation", time_range=10,
                                     config="unused", num_epoch=1, processed_root=None, query_time="on")
        self.args = args
        self.start_training = Mock(side_effect=RuntimeError("validation completed"))
        scope = {
            "os": os, "PROJECT_ROOT": str(root), "parse_args": lambda: args,
            "util": types.SimpleNamespace(load_config=Mock(return_value=cfg),
                                           create_working_directory=self.start_training),
            # Deliberately omit protocol helpers from the training module.
            "pretrain": types.SimpleNamespace(),
        }
        source = Path(__file__).resolve().parents[1] / "script" / "train.py"
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body
                 if (isinstance(node, ast.ImportFrom) and node.module in ("pine.data.protocol", "pine.cache_validation"))
                 or (isinstance(node, ast.FunctionDef) and node.name == "main")]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
        self.main = scope["main"]

    def test_valid_data_reaches_training_without_pretrain_protocol_helpers(self):
        with self.assertRaisesRegex(RuntimeError, "validation completed"):
            self.main()
        self.start_training.assert_called_once()

    def test_overlapping_splits_stop_before_training(self):
        (self.raw / "valid.txt").write_text((self.raw / "train.txt").read_text())
        with self.assertRaisesRegex(ValueError, "train/valid overlap"):
            self.main()
        self.start_training.assert_not_called()

    def test_stale_processed_data_stops_before_training(self):
        (self.raw / "train.txt").write_text("0\t0\t1\t5\n")
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.main()
        self.start_training.assert_not_called()

    def test_processed_root_override_uses_new_artifacts(self):
        new_root = self.raw.parents[2] / "fresh_processed"
        new_processed = new_root / "interpolation" / "fixture"
        new_processed.mkdir(parents=True)
        marker = self.processed / "storage_format.json"
        (new_processed / marker.name).write_text(marker.read_text())
        marker.write_text(json.dumps({"protocol": "old", "setting": "interpolation"}))
        self.args.processed_root = str(new_root)
        with self.assertRaisesRegex(RuntimeError, "validation completed"):
            self.main()
        self.start_training.assert_called_once()

    def test_protocol_error_reports_artifact_path_and_actual_version(self):
        marker = self.processed / "storage_format.json"
        marker.write_text(json.dumps({"protocol": "old", "setting": "interpolation"}))
        with self.assertRaises(ValueError) as caught:
            self.main()
        message = str(caught.exception)
        self.assertIn(str(marker), message)
        self.assertIn("found protocol='old'", message)
        self.assertIn(f"expected protocol='{PROTOCOL}'", message)
        self.start_training.assert_not_called()


if __name__ == "__main__":
    unittest.main()
