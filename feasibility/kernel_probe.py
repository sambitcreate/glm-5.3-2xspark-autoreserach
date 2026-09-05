"""Bounded synthetic E2 compile/correctness/timing probe; NOT a promotion gate."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '12.1a')
os.environ.setdefault('MAX_JOBS', '2')
import torch
from torch.utils.cpp_extension import load
import exllamav3_ext as reference

root = Path(__file__).resolve().parent
outdir = Path(os.environ.get('PROBE_ARTIFACT_DIR', str(root.parent / 'artifacts/feasibility')))
outdir.mkdir(parents=True, exist_ok=True)
source = root / 'ext-source/quant/exl3_fat_gemm.cu'
original = source.read_text()
assert original.count('__launch_bounds__(FAT_THREADS)') == 1
binding = root / 'bindings.cpp'
binding.write_text('''#include <torch/extension.h>
#include "ext-source/quant/exl3_fat_gemm.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
 m.def("direct", &exl3_fat_gemm);
 m.def("scatter", &exl3_fat_gemm_scatter);
}
''')
results = {'scope': 'head-only synthetic standalone E2 extension; no model inference candidate deployment',
           'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'original_sha256': hashlib.sha256(original.encode()).hexdigest(), 'variants': []}

def evaluate(mod):
    torch.manual_seed(41)
    rows = []
    # Actual TP2 dimensions: gate/up combined K4096,N2048; down K1024,N4096.
    for k,n in ((256,256), (4096,2048), (1024,4096)):
        packed = torch.randint(-30000,30000,(k//16,n//16,64),dtype=torch.int16,device='cuda')
        svh = torch.ones(n,dtype=torch.float16,device='cuda')
        for m in (1,127,128,129,145,256,512,2048):
            a = torch.randn(m,k,dtype=torch.float16,device='cuda') * 0.1
            ref = torch.empty(m,n,device='cuda',dtype=torch.float32)
            got = torch.empty_like(ref)
            reference.exl3_fat_gemm(a,packed,ref,svh,4,True,False)
            mod.direct(a,packed,got,svh,4,True,False)
            torch.testing.assert_close(got,ref,atol=1e-4,rtol=1e-5)
            assert torch.isfinite(got).all()
            idx = torch.randperm(m+17,device='cuda')[:m].contiguous()
            weight = torch.rand(m,device='cuda',dtype=torch.float16)
            init = torch.randn(m+17,n,device='cuda',dtype=torch.float32)*0.01
            sr,sg = init.clone(),init.clone()
            reference.exl3_fat_gemm_scatter(a,packed,sr,svh,idx,weight,4,True,False)
            mod.scatter(a,packed,sg,svh,idx,weight,4,True,False)
            torch.testing.assert_close(sg,sr,atol=1e-4,rtol=1e-5)
            # Ensure a deliberately invalid numerical output cannot pass this gate.
            bad = got.clone(); bad[0,0] += 100
            try:
                torch.testing.assert_close(bad,ref,atol=1e-4,rtol=1e-5)
            except AssertionError:
                pass
            else:
                raise AssertionError('negative control was accepted')
            for _ in range(3):
                mod.direct(a,packed,got,svh,4,True,False)
            samples=[]
            for _ in range(7):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10): mod.direct(a,packed,got,svh,4,True,False)
                end.record();end.synchronize()
                samples.append(start.elapsed_time(end)/10)
            rows.append({'m':m,'k':k,'n':n,'direct_ms':statistics.median(samples),'samples_ms':samples,
                         'direct_maxabs':float((got-ref).abs().max()),'scatter_maxabs':float((sg-sr).abs().max())})
    # Graph replay smoke, distinct from full serving graph capture.
    graph=torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph): mod.direct(a,packed,got,svh,4,True,False)
    graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(got,ref,atol=1e-4,rtol=1e-5)
    return rows

for name, text in [('baseline',original),('minblocks2',original.replace('__launch_bounds__(FAT_THREADS)', '__launch_bounds__(FAT_THREADS, 2)'))]:
    candidate=source.with_name(f'probe_{name}.cu');candidate.write_text(text)
    build=root/'build'/name;build.mkdir(parents=True,exist_ok=True)
    t=time.monotonic()
    mod=load(name=f'glm_e2_probe_{name}',sources=[str(binding),str(candidate)],build_directory=str(build),
             extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)
    rec={'name':name,'build_seconds':time.monotonic()-t,'source_sha256':hashlib.sha256(text.encode()).hexdigest(),
         'binary_sha256':hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()}
    rec['cases']=evaluate(mod)
    rec['correctness']='pass';rec['negative_control']='rejected';rec['graph_smoke']='pass'
    results['variants'].append(rec)
    (outdir/'kernel-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('PROBE',name,'build_seconds',rec['build_seconds'],'cases',len(rec['cases']),flush=True)
base,cand=results['variants']
ratios=[b['direct_ms']/c['direct_ms'] for b,c in zip(base['cases'],cand['cases'])]
import math
results['synthetic_geomean_speedup']=math.exp(statistics.mean(map(math.log,ratios)))
results['decision']='feasibility_only_no_promotion'
results['limitations']=['sequential variant timing without interleaved A/B','synthetic cache-warm weights',
 'no full MoE pipeline timing','no candidate two-rank inference','no sanitizer','not a statistical performance claim']
(outdir/'kernel-results.json').write_text(json.dumps(results,indent=2)+'\n')
print('FEASIBILITY_COMPLETE',results['synthetic_geomean_speedup'],results['decision'],flush=True)
