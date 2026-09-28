import argparse
import importlib.machinery
import importlib.util
import inspect
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('library', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
path = str(args.library.resolve())
loader = importlib.machinery.ExtensionFileLoader('mistralrs', path)
spec = importlib.util.spec_from_file_location('mistralrs', path, loader=loader)
module = importlib.util.module_from_spec(spec)
loader.exec_module(module)
for name in [None, 'isaac64', 'keyed-threefry2x32-v1']:
    module.CompletionRequest(prompt='Hello', model='model', sampling_rng=name)
    module.ChatCompletionRequest(messages=[{'role': 'user', 'content': 'Hello'}], model='model', sampling_rng=name)
for cls, kwargs in [(module.CompletionRequest, {'prompt': 'Hello'}), (module.ChatCompletionRequest, {'messages': [{'role': 'user', 'content': 'Hello'}]})]:
    try:
        cls(model='model', sampling_rng='typo', **kwargs)
    except ValueError as error:
        assert 'Unknown sampling RNG' in str(error)
    else:
        raise AssertionError('invalid RNG accepted')
    assert inspect.signature(cls).parameters['sampling_rng'].default is None
assert inspect.signature(module.Runner).parameters['sampling_rng'].default == 'isaac64'
result = {'passed': True, 'module': path, 'request_choices': [None, 'isaac64', 'keyed-threefry2x32-v1'], 'request_types': ['CompletionRequest', 'ChatCompletionRequest'], 'unknown_rng_rejected': True, 'runner_signature': module.Runner.__text_signature__}
args.output.write_text(json.dumps(result, indent=2) + '\n')
print('PASS: Python RNG parameters, inheritance, invalid-value rejection, and Runner default')
