import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'feasibility/check_weights.py'


class WeightValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snap = self.root / 'hub/repo/snapshots/test'
        self.snap.mkdir(parents=True)
        blob = self.root / 'hub/repo/blobs/data'
        blob.parent.mkdir()
        blob.write_bytes(b'fixture shard; not real model data')
        (self.snap / 'model.safetensors').symlink_to('../../blobs/data')
        (self.snap / 'config.json').write_text('{"hidden_size":256}')
        (self.snap / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'model.safetensors'}}))

    def run_helper(self):
        env = dict(os.environ, MODEL_DIR=str(self.snap), DFLASH_MODEL_DIR=str(self.snap))
        return subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)

    def test_valid_snapshot_and_unchanged_files(self):
        before = {str(p): p.read_bytes() for p in self.snap.iterdir() if p.is_file()}
        r = self.run_helper()
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual(data['MODEL_DIR']['shards'], 1)
        self.assertEqual(data['MODEL_DIR']['snapshot'], str(self.snap))
        self.assertEqual(data['MODEL_DIR'], data['DFLASH_MODEL_DIR'])
        after = {str(p): p.read_bytes() for p in self.snap.iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_blob_is_failure(self):
        (self.root / 'hub/repo/blobs/data').unlink()
        self.assertNotEqual(self.run_helper().returncode, 0)

    def test_missing_indexed_shard_is_failure(self):
        (self.snap / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'missing.safetensors'}}))
        self.assertNotEqual(self.run_helper().returncode, 0)

    def test_unindexed_draft_supported(self):
        (self.snap / 'model.safetensors.index.json').unlink()
        r = self.run_helper()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)['DFLASH_MODEL_DIR']['shards'], 1)

    def test_empty_shard_is_failure(self):
        (self.root / 'hub/repo/blobs/data').write_bytes(b'')
        self.assertNotEqual(self.run_helper().returncode, 0)


if __name__ == '__main__':
    unittest.main()
