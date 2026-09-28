import argparse
import hashlib
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('binary', type=Path)
parser.add_argument('model', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
layers = json.loads((args.model / 'config.json').read_text())['num_hidden_layers']
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
names = ['isaac64', 'keyed-threefry2x32-v1']
results = []
texts = {}
for default in names:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    command = [str(args.binary.resolve()), 'serve', '-m', str(args.model.resolve()), '--host', '127.0.0.1', '--port', str(port), '--no-ui', '--dtype', 'f32', '--format', 'plain', '--token-source', 'none', '--paged-attn', 'off', '--device-layers', str(layers), '--sampling-rng', default]
    print('START server', default, flush=True)
    with (args.output / f'http-{default}.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited: {default}')
                try:
                    with opener.open(base + '/v1/models', timeout=1) as response:
                        assert response.status == 200
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.2)
            for seed in [42, 123]:
                for override in [None, *names]:
                    payload = {'model': 'default', 'prompt': 'Once upon a time, a curious cat', 'seed': seed, 'temperature': 0.8, 'top_k': 40, 'top_p': 0.9, 'max_tokens': 24, 'ignore_eos': True}
                    if override is not None:
                        payload['sampling_rng'] = override
                    request = urllib.request.Request(base + '/v1/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
                    with opener.open(request, timeout=30) as response:
                        body = json.load(response)
                    text = body['choices'][0]['text']
                    texts[default, override, seed] = text
                    results.append({'engine_rng': default, 'request_rng': override, 'seed': seed, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'text': text})
            payload['sampling_rng'] = 'typo'
            request = urllib.request.Request(base + '/v1/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            try:
                opener.open(request, timeout=5)
            except urllib.error.HTTPError as error:
                assert error.code in [400, 422], error.code
            else:
                raise AssertionError('invalid RNG was accepted')
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    print('FINISH server', default, flush=True)
for seed in [42, 123]:
    for name in names:
        expected = texts[name, None, seed]
        assert texts[names[0], name, seed] == expected
        assert texts[names[1], name, seed] == expected
assert any(texts[names[0], None, seed] != texts[names[1], None, seed] for seed in [42, 123])
summary = {'passed': True, 'cases': results, 'checks': ['omitted request RNG inherits engine default', 'explicit request RNG overrides either engine default', 'both algorithms reproduce their seeded output across engine defaults', 'algorithms produce distinct outputs on this fixture', 'unknown request RNG rejects with HTTP 400/422']}
(args.output / 'http-validation.json').write_text(json.dumps(summary, indent=2) + '\n')
print('PASS: request overrides and inheritance across both engine RNG defaults', flush=True)
