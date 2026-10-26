"""Exercise the real preprocessing control flow with graph construction stubbed."""

import ast
import json
import tempfile
import types
import unittest
from pathlib import Path

from pine.data.protocol import (PROTOCOL, iter_contexts, read_splits, source_fingerprints,
                               validate_processed, validate_splits)


class PreprocessingContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.raw, self.output = root / "raw", root / "processed"
        self.raw.mkdir()
        self.output.mkdir()
        rows = {"train": "0\t0\t1\t0\n", "msg": "0\t0\t1\t1\n0\t0\t1\t4\n",
                "valid": "0\t0\t1\t2\n", "test": "0\t0\t1\t3\n"}
        for split, contents in rows.items():
            (self.raw / f"{split}.txt").write_text(contents)
        self.saved = {}

        def make_graph(rows, **metadata):
            return types.SimpleNamespace(rows=rows, num_edges=2 * len(rows),
                                         relation_graph=types.SimpleNamespace(num_edges=0))

        def save(graph, path):
            self.saved[Path(path).relative_to(self.output).as_posix()] = graph.rows
            Path(path).write_bytes(b"graph I/O stub")

        self.scope = {"json": json, "torch": types.SimpleNamespace(save=save),
                      "PROTOCOL": PROTOCOL, "iter_contexts": iter_contexts,
                      "read_splits": read_splits, "source_fingerprints": source_fingerprints,
                      "validate_splits": validate_splits, "make_graph": make_graph,
                      "SHARDED_MSG_DATASETS": {"ICEWS14", "ICEWS05-15"}}
        self.load("pine/data/shards.py", "write_shards")
        self.load("pine/preprocess.py", "write_storage_marker")
        self.load("pine/preprocess.py", "process_mixed_dataset")

    def load(self, relative_path, name):
        path = Path(__file__).resolve().parents[1] / relative_path
        node = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), self.scope)

    def build(self, setting, dataset):
        storage = self.scope["process_mixed_dataset"](
            self.raw, self.output, dataset, setting, 30, 60, 5000, shard_msg=False)
        self.scope["write_storage_marker"](
            self.output, {"train": "sharded", "msg": storage}, dataset, setting, self.raw)
        return storage, validate_processed(self.raw, self.output, setting)

    def test_extrapolation_automatically_writes_verified_causal_message_shards(self):
        storage, marker = self.build("extrapolation", "icews21")
        self.assertEqual(storage, "sharded")
        self.assertEqual(marker["protocol"], PROTOCOL)
        self.assertEqual(marker["setting"], "extrapolation")
        self.assertEqual(marker["source_sha256"], source_fingerprints(self.raw))
        manifest = json.loads((self.output / "msg_manifest.json").read_text())
        self.assertTrue(manifest["causal_evaluation"])
        self.assertEqual(manifest["protocol"], PROTOCOL)
        self.assertEqual(manifest["time_to_file"].keys(), {"2", "3"})
        for shard in manifest["shards"]:
            self.assertTrue(all(row[3] < shard["query_time_min"] for row in self.saved[shard["file"]]))

    def test_interpolation_monolithic_message_graph_keeps_future_support(self):
        storage, marker = self.build("interpolation", "GDELT")
        self.assertEqual(storage, "monolithic")
        self.assertEqual(marker["setting"], "interpolation")
        self.assertEqual([row[3] for row in self.saved["msg.pt"]], [1, 4])

    def test_rebuild_replaces_legacy_marker(self):
        (self.output / "storage_format.json").write_text(json.dumps({"format": "pine_storage_v1"}))
        _, marker = self.build("extrapolation", "icews21")
        self.assertEqual(marker["protocol"], PROTOCOL)
        self.assertEqual(marker["storage"]["msg"], "sharded")


if __name__ == "__main__":
    unittest.main()
