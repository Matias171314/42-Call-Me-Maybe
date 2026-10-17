*This project has been created as part of the 42 curriculum by mvasquez.*

# Call Me Maybe

## Description

Convert natural-language requests into function calls using Qwen/Qwen3-0.6B
and the supplied LLM SDK. The program selects a function and extracts its input
arguments; it does not execute the function.

Generation is constrained token by token. Function names must belong to the
provided definitions, numbers must follow JSON syntax, strings must have valid
escapes, and booleans are restricted to `true` and `false`.

## Instructions

Python 3.10+ and `uv` are required. The first model download needs network access.
The supplied `llm_sdk` directory must remain alongside `src`.

```sh
make install
make run
make debug
make lint
make lint-strict
make clean
```

Custom input and output paths:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

The same options can be passed through Make:

```sh
make run ARGS="--functions_definition /tmp/functions.json --input /tmp/prompts.json --output /tmp/results.json"
```

By default, the program uses the paths shown in the first example. It creates
missing output directories. Input and processing errors are reported with a
nonzero exit code. An empty prompt list produces `[]` without loading the model.
Thread settings are left to the runtime; the application does not modify them.

The lint targets check the project and exclude the supplied SDK and virtual
environment. Testing tools and the original moulinette belong in the separate
companion directory, not in the submission.

## Input and output

Both inputs are JSON arrays. A prompt entry requires a string field `prompt`.
A function definition has `name`, `description`, `parameters` and `returns`.
Each parameter and return definition contains `type`. Argument generation
supports `string`, `number`, `integer` and `boolean`.

Every output entry has exactly these keys:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The output follows the supplied subject and moulinette (`name`, `parameters`).
The correction sheet uses different names (`fn_name`, `args`).

## Implementation

- `src/__main__.py`: parse options, read inputs, load the model, save results.
- `src/parsing/`: validate input records using Pydantic.
- `src/model_engine.py`: use the SDK's public inference methods and cache decoded
  tokens. The SDK is imported only when a model is actually needed.
- `src/function_calling_engine.py`: select a function and generate each argument
  using its definition and the original request. For strings, the model chooses
  between copying a literal and deriving a value. Literal candidates are exact
  word or quoted spans from the request; derived values use normal typed
  generation. There are no branches based on parameter names.
- `src/constrained_decoder.py`: choose the highest-logit token, mask it to
  negative infinity if invalid, and try the next. Stop at a completed value or
  raise an error when no valid completion is available within the token limit.

JSON keys and punctuation are constructed by Python. Numeric prefixes are
checked step by step and completed numbers are checked with `json.loads`.
String generation validates escapes and detects unescaped closing quotes.
Integer conversion preserves precision; nonfinite floating-point values are
rejected. No regular expressions are used by the decoder.

The engine does not contain replacement tables, path-specific extraction,
post-generation quote repairs or forced retries after `NO_MATCH`. Unsupported
requests produce a clear error. Formal values implied by a request are inferred
by the model, rather than selected from hardcoded examples.

The decoder guarantees syntax and primitive types for successful outputs, not
correct interpretation of every request. The small model can still choose an
incorrect function or argument.

Pydantic validates the input records in the parsing modules. The model adapter,
function-calling engine and constrained decoder are ordinary Python classes.
The supplied SDK is unchanged, and application code does not import its internal
machine-learning dependencies or use its private attributes.

## Validation

The original, unmodified moulinette was run from the separate testing directory.
Both runs used this machine's normal thread settings, without OMP/MKL overrides,
on CPU with a cached model. Times include model loading.

| Set | Correct calls | Time |
| --- | --- | --- |
| Private | 11/11 (100%) | 117.84 seconds |
| Public | 9/11 (81.8%) | 101.72 seconds |
| Combined | 20/22 (90.9%) | 219.56 seconds |

All private function names and argument values were also checked against the
provided corrections and matched exactly. All 22 outputs passed independent
checks for exact keys, required arguments and primitive types.

Two public requests still fail when inferring patterns or symbols. The public
subset alone remains below 90%; the combined accuracy exceeds 90%. These are
semantic model errors, not invalid JSON. They have not been patched with fixed
answers or parameter-name rules.

The final local checks pass 21 regression tests, flake8 and mypy in strict mode.
A clean copy installs with `uv sync --locked` in a fresh virtual environment and
supports the documented CLI. All test programs, logs and grading files remain
outside the delivery directory.

## Resources

- [Python JSON documentation](https://docs.python.org/3/library/json.html)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)
- [NumPy argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)

AI assistance was used to review requirements, compare implementations, refactor
code and run external checks. The student should understand and be able to
modify the decoding logic during the defense.
