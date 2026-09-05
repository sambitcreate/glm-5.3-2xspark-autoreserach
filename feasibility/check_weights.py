"""Read-only validation of the serving container's selected HF snapshots."""
import hashlib
import json
import os
from pathlib import Path

result = {}
for key in ('MODEL_DIR', 'DFLASH_MODEL_DIR'):
    root = Path(os.environ[key])
    assert root.is_dir(), root
    indices = sorted(root.glob('*.safetensors.index.json'))
    if indices:
        shards = sorted({s for p in indices for s in json.loads(p.read_text())['weight_map'].values()})
    else:
        shards = sorted(p.name for p in root.glob('*.safetensors'))
    assert shards, f'No shards: {root}'
    entries = []
    for name in shards:
        p = root / name
        assert p.is_file() and p.stat().st_size > 0, p
        entries.append({'name': name, 'resolved': str(p.resolve()), 'bytes': p.stat().st_size})
    config = root / 'config.json'
    cfg = json.loads(config.read_text())
    result[key] = {'snapshot': str(root), 'shards': len(shards), 'total_bytes': sum(x['bytes'] for x in entries),
                   'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                   'layout_sha256': hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
                   'model_dimensions': {k:v for k,v in cfg.items() if k in ('hidden_size','moe_intermediate_size','num_experts','n_routed_experts','num_hidden_layers')},
                   'verification': 'config hash and all indexed shard symlink targets/sizes; not full shard content hashes'}
print(json.dumps(result, indent=2))
