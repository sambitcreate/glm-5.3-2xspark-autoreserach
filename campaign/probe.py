"""Time-limited, paired synthetic kernel screening. No service patching."""
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import torch
import exllamav3_ext as ref
from routing import baseline as route_base, candidate as route_candidate

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/timed-campaign'
DEADLINE = json.loads((ART/'deadline.json').read_text())['deadline_epoch'] - 660
REPORT = ROOT/'reports/timed-30min.md'
HOLDOUT = os.environ.get('CAMPAIGN_HOLDOUT') == '1'
records=[]

def check_time():
    if time.time() >= DEADLINE or (ART/'STOP').exists():
        raise TimeoutError('Experiment window ended; reserved 11 minutes for restoration')

def record(name, data):
    data={'experiment':name, 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), **data}
    records.append(data)
    (ART/('holdout-results.json' if HOLDOUT else 'kernel-results.json')).write_text(json.dumps(records,indent=2)+'\n')
    summary=data.get('decision',data.get('status','recorded'))
    if 'speedup' in data: summary+=f"; paired synthetic ratio={data['speedup']:.4f}x"
    with REPORT.open('a') as f:f.write(f"| {data['utc'][11:19]} | {name} | {summary} |\n")
    print(name,summary,flush=True)

def paired(base, cand, inner=30):
    for fn in (base,cand):
        for _ in range(5):fn()
    torch.cuda.synchronize()
    pairs=[]
    for j in range(15):
        check_time()
        values={}
        for label,fn in ([('b',base),('c',cand)] if j%2==0 else [('c',cand),('b',base)]):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(inner):fn()
            end.record();end.synchronize()
            values[label]=start.elapsed_time(end)/inner
        pairs.append(values)
    return {'baseline_ms':statistics.median(p['b'] for p in pairs),
            'candidate_ms':statistics.median(p['c'] for p in pairs),
            'speedup':statistics.median(p['b']/p['c'] for p in pairs), 'pairs':pairs}


def graph_fn(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): result=fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): result=fn()
    return graph.replay,result


def load_probe(name):
    path=ROOT/'feasibility/build'/name/f'glm_e2_probe_{name}.so'
    spec=importlib.util.spec_from_file_location(f'glm_e2_probe_{name}',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

try:
    torch.manual_seed(2027 if HOLDOUT else 2026)
    base=load_probe('baseline');candidate=load_probe('minblocks2')
    # Repeat real-sized fat shapes, including prior regressions, AB/BA ordering.
    for k,n in ((4096,2048),(1024,4096)):
        packed=torch.randint(-30000,30000,(k//16,n//16,64),dtype=torch.int16,device='cuda')
        svh=torch.ones(n,device='cuda',dtype=torch.float16)
        for m in ((193,384,768,1536,3072,4096,6144) if HOLDOUT else (129,145,256,512,1024,2048,7168)):
            check_time()
            a=torch.randn(m,k,device='cuda',dtype=torch.float16)*.1
            b=torch.empty(m,n,device='cuda',dtype=torch.float32);c=torch.empty_like(b)
            from prefill_dispatch import use_minblocks2
            selected = candidate if not HOLDOUT or use_minblocks2(m,k,n) else base
            base.direct(a,packed,b,svh,4,True,False);selected.direct(a,packed,c,svh,4,True,False)
            torch.testing.assert_close(b,c,atol=1e-4,rtol=1e-5)
            timings=paired(lambda:base.direct(a,packed,b,svh,4,True,False),
                          lambda:selected.direct(a,packed,c,svh,4,True,False),inner=10)
            record(f'{"holdout-" if HOLDOUT else ""}prefill-direct-M{m}-K{k}-N{n}',{'shape':[m,k,n], 'selected':'minblocks2' if selected is candidate else 'baseline', 'maxabs':float((b-c).abs().max()),
                   **timings,'decision':'screen-only; no TTFT claim'})
    # Include mapped sentinel n_exp and non-uniform weights; sentinel gets a count slot.
    for tokens in ((2,4,24,64) if HOLDOUT else (1,8,16,32,128,512,2048)):
        check_time()
        ids=torch.rand(tokens,288,device='cuda').topk(8,dim=-1).indices.contiguous()
        if tokens>1:ids[0,0]=288
        weights=torch.rand(tokens,8,device='cuda',dtype=torch.float32)
        expected=route_base(ids,weights,288);got=route_candidate(ids,weights,288)
        for b,c in zip(expected,got):torch.testing.assert_close(b,c,atol=0,rtol=0)
        eager=paired(lambda:route_base(ids,weights,288),lambda:route_candidate(ids,weights,288))
        bg,bout=graph_fn(lambda:route_base(ids,weights,288))
        cg,cout=graph_fn(lambda:route_candidate(ids,weights,288))
        bg();cg();torch.cuda.synchronize()
        for b,c in zip(bout,cout):torch.testing.assert_close(b,c,atol=0,rtol=0)
        graph=paired(bg,cg,inner=100)
        # Negative control: changing a packed token must fail equality.
        damaged=got[0].clone();damaged[0]+=1
        assert not torch.equal(damaged,expected[0])
        record(f'decode-routing-T{tokens}',{'tokens':tokens,'correctness':'exact','graph_parity':'exact',
               'eager':eager,'graph':graph,'speedup':graph['speedup'],
               'decision':'screen-only; routing metadata, not end-to-end decode'})
    # Full routed MoE call on a reduced-expert synthetic layer, same TP-local
    # dimensions. This checks whether metadata savings survive actual GEMM work.
    check_time()
    import inspect
    from vllm.model_executor.layers.quantization import exl3 as overlay
    helper_path=Path('/opt/glm53/test_exl3_overlay.py')
    spec=importlib.util.spec_from_file_location('stock_selfcheck',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    n_exp = 288 if HOLDOUT else 16
    _,layer=helper._tiny_layer(torch.device('cuda'),n_exp=n_exp,hidden=4096,inter=1024)
    original=inspect.getsource(overlay.apply_exl3_fused_moe)
    start=original.index('    flat_token = ')
    end=original.index('    out = torch.zeros',start)
    changed=original[:start]+('    token_sorted, weight_sorted, expert_count = '
        '_route_candidate(local.reshape(tokens, topk), weights.contiguous(), n_exp)\n')+original[end:]
    namespace=dict(vars(overlay));namespace['_route_candidate']=route_candidate
    exec(compile(changed,'<experimental-route-pack>','exec'),namespace)
    optimized=namespace['apply_exl3_fused_moe']
    (ART/'experimental-moe.py').write_text(changed)
    for tokens in (1,8,16,32):
        check_time()
        x=torch.randn(tokens,4096,device='cuda',dtype=torch.float16)*.01
        ids=torch.rand(tokens,n_exp,device='cuda').topk(8,dim=-1).indices.contiguous()
        weights=torch.rand(tokens,8,device='cuda',dtype=torch.float32)
        weights=weights/weights.sum(-1,keepdim=True)
        args=(x,ids,weights,layer,layer._exl3_inners,None,10.0)
        b=overlay.apply_exl3_fused_moe(*args);c=optimized(*args)
        torch.testing.assert_close(b,c,atol=1e-4,rtol=1e-5)
        bg,bout=graph_fn(lambda:overlay.apply_exl3_fused_moe(*args))
        cg,cout=graph_fn(lambda:optimized(*args))
        bg();cg();torch.cuda.synchronize()
        torch.testing.assert_close(bout,cout,atol=1e-4,rtol=1e-5)
        timing=paired(bg,cg,inner=30)
        record(f'{"holdout-" if HOLDOUT else ""}decode-full-moe-T{tokens}',{'tokens':tokens,'experts':n_exp,'hidden':4096,'intermediate':1024,
               'maxabs':float((b-c).abs().max()),**timing,
               'decision':f'{n_exp}-expert synthetic full MoE; no serving claim'})
    record('campaign-kernel-screen',{'status':'passed','decision':'No promotion; restore known-good service'})
except Exception as exc:
    record('campaign-stopped',{'status':'failed' if not isinstance(exc,TimeoutError) else 'budget-stop',
           'error':f'{type(exc).__name__}: {exc}', 'decision':'restore known-good service'})
    raise
