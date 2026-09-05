"""Run inside the existing head container; never print its API credential."""
import json
import os
import time
import urllib.request

base = os.environ.get('PROBE_API_URL', 'http://127.0.0.1:8888').rstrip('/')
deadline = time.monotonic() + int(os.environ.get('PROBE_READY_TIMEOUT', '3600'))
while True:
    try:
        with urllib.request.urlopen(base + '/health', timeout=10) as r:
            assert r.status == 200
        break
    except Exception:
        if time.monotonic() >= deadline:
            raise RuntimeError('Service did not become healthy before deadline')
        time.sleep(5)
headers = {'Content-Type': 'application/json'}
key = os.environ.get('VLLM_API_KEY')
if key:
    headers['Authorization'] = 'Bearer ' + key
with urllib.request.urlopen(urllib.request.Request(base+'/v1/models', headers=headers), timeout=30) as r:
    models=json.load(r)
model=models['data'][0]['id']
body={'model':model,'messages':[{'role':'user','content':'Reply with exactly OK.'}], 'temperature':0,
      'max_tokens':16,'chat_template_kwargs':{'enable_thinking':False}}
req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers=headers)
with urllib.request.urlopen(req,timeout=180) as r:
    obj=json.load(r)
text=obj['choices'][0]['message'].get('content','')
assert text and 'OK' in text, 'Smoke response did not contain OK'
print(json.dumps({'health':200,'model':model,'content':text,'usage':obj.get('usage'),
                  'scope':'service restoration smoke, not quality validation'}))
