"""Small real-service baseline only; does not install candidate kernels."""
import json,os,time,uuid,urllib.request
base=os.environ.get('PROBE_API_URL','http://127.0.0.1:8888')
headers={'Content-Type':'application/json'}
if os.environ.get('VLLM_API_KEY'):headers['Authorization']='Bearer '+os.environ['VLLM_API_KEY']
with urllib.request.urlopen(urllib.request.Request(base+'/v1/models',headers=headers),timeout=15) as r:model=json.load(r)['data'][0]['id']

def run(label,prompt,max_tokens):
    body={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':max_tokens,
          'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
    t=time.perf_counter();first=None;usage=None;text=''
    with urllib.request.urlopen(urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers=headers),timeout=150) as r:
        for line in r:
            if not line.startswith(b'data:'):continue
            raw=line[5:].strip()
            if raw==b'[DONE]':continue
            obj=json.loads(raw)
            if obj.get('usage'):usage=obj['usage']
            for choice in obj.get('choices',[]):
                content=choice.get('delta',{}).get('content') or ''
                if content:
                    if first is None:first=time.perf_counter()
                    text+=content
    end=time.perf_counter()
    assert first is not None and usage and text,'Incomplete response'
    return {'label':label,'ttft_s':first-t,'decode_tok_s':(usage['completion_tokens']-1)/(end-first),
            'usage':usage,'wall_s':end-t,'output_preview':text[:80],
            'note':'client timing; service baseline only, not a candidate comparison'}

out=[]
out.append(run('unique-salt-approx8k',str(uuid.uuid4())+'\n'+'the '*8000+'\nReply with OK.',8))
for i in range(3):
    out.append(run('structured-'+str(i),'Count from 1 to 200. Output only the numbers, separated by spaces.',128))
    out.append(run('prose-'+str(i),'Explain hash maps, collision handling and resizing in detail.',128))
print(json.dumps({'model':model,'runs':out,'limitations':['3 short decode samples','uncached status not independently gated','no candidate service comparison']},indent=2))
