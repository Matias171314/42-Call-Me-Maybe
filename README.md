*This project has been created as part of the 42 curriculum by mvasquez.*

# Call Me Maybe

## Description

Translate natural-language requests into function names and typed arguments with
Qwen/Qwen3-0.6B. The program produces calls; it does not execute the supplied
functions, SQL queries, or file operations. Only the supplied LLM SDK loads the
model. Application code uses its public encode, decode and logits methods.

## Instructions

Python 3.10+ and uv are required. The first run downloads the model and therefore
requires network access and enough disk space. CPU inference uses four threads
by default; override OMP_NUM_THREADS and MKL_NUM_THREADS before launching if
needed. The SDK automatically chooses available GPU acceleration.

```sh
make install
make run
uv run python -m src --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
make debug
make lint
make lint-strict
```

The supplied moulinette can be run separately: prepare an exercise set, pass its
input files to the command above, then grade the generated output. During
development, regression tests, evaluation tooling and grading results were kept
in a separate companion folder outside the submission repository.

## Input and output

Both inputs are JSON arrays. Prompt entries contain exactly `prompt` (a string).
Function entries contain `name`, `description`, `parameters` and `returns`.
Parameters support `string`, `number`, `integer` and `boolean`, with an optional
`description` and nonempty `enum`. Unsupported argument types, duplicate names,
malformed JSON and missing files produce a clear error and a nonzero exit code.
An empty prompt list produces an empty result array. Requests that the model
identifies as unsupported produce a clear error rather than a fabricated call.

Output contains exactly `prompt`, `name` and `parameters` per entry, preserving
input order. These are the keys required by the subject and the attached
moulinette. The evaluation sheet's `fn_name`/`args` wording conflicts with both.

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

## Algorithm and design decisions

1. Validate inputs with strict Pydantic models before loading the LLM.
2. Ask the model to choose a function from descriptions. During every generation
   step, reject token continuations that cannot complete an available name.
   Shared name prefixes remain open until a longer name or terminator is chosen.
   Once only one candidate remains, its suffix is forced without further model
   calls. Unexpected no-match decisions receive a second LLM applicability check.
3. Constrain path arguments to literal path spans present in the request,
   including quoted paths with spaces and Windows separators. The model chooses
   among these spans; there are no exercise-specific paths or answers.
4. Extract each argument with its function description, full request and preceding
   arguments. JSON punctuation and keys are constructed by the program.
5. Constrain numeric prefixes, JSON string escapes and boolean/enum candidates
   before accepting each token. Strings stop at an unescaped closing quote.
6. Recover omitted embedded quotes when the generated text matches a literal
   after a colon, trim unquoted path padding and resolve named replacement
   characters through the Unicode database. These bounded argument recoveries
   never select a function or load reference answers.
7. Validate argument types, reject nonfinite numbers and serialize with `json`,
   disabling NaN/Infinity. Decoder exhaustion is reported rather than fabricating
   arguments.

The valid token with the highest logit is selected deterministically. NumPy
partial ranking examines top candidates first and expands to the entire
vocabulary only when necessary. This is equivalent to greedy selection after
masking invalid tokens. Empty special-token decodings are excluded so they
cannot consume the generation budget without progress.

The model wrapper, decoder, engine and input records use Pydantic. Exceptions
remain ordinary Python exceptions. The SDK is provided unchanged. No application
code imports torch, transformers or accesses private SDK attributes.

## Performance and reliability

Constrained decoding guarantees the syntax of successfully emitted values;
it cannot guarantee semantic accuracy. A model can choose the wrong function
or extract a syntactically valid but incorrect argument. The external evaluation runner measures
both the actual score and elapsed time on the current machine.

Measured on this machine (Intel i7-11800H, CPU, four threads, cached model):

| Supplied moulinette set | Score | Time including model loading |
| --- | --- | --- |
| Public | 11/11 (100%) | 81.55 seconds |
| Private | 11/11 (100%) | 83.75 seconds |

All 22 outputs passed independent checks for exact keys, required arguments and
primitive types. These measurements come from separate runs of each set.
The final checks passed 13 regression tests, flake8 and mypy in strict mode.
Development grading results are stored in the separate testing folder. Scores describe these supplied tests, not all possible prompts.

Working bonuses: named-character and literal recovery, a second model check
for unexpected no-match decisions, and decoded vocabulary tokens cached per
model instance, reducing repeated tokenizer work. Partial logit ranking and short extraction contexts
reduce overhead. The supplied SDK evaluates the full context at each step;
there is no KV cache in its public API. Longer prompts and CPU memory pressure
therefore increase runtime. Progress is displayed once per completed prompt.

## Challenges and testing strategy

The initial implementation did not support integer arguments used by private
exercises, rejected empty strings, repeatedly sorted the full vocabulary and
used long extraction instructions. The refactor adds integer validation, accepts
empty strings, caches decoding and reduces prompt length. Escaped quotes and
Windows paths are treated as JSON string content rather than split on quotes.

Regression tests use an adversarial fake SDK to verify token masking, empty
special tokens, shared function prefixes, negative decimals, empty strings,
JSON escapes and malformed input. Real end-to-end testing uses the original
moulinette public and private exercise sets. `make lint` and `make lint-strict`
check application code; the supplied SDK is excluded.

## Resources

- [Python JSON documentation](https://docs.python.org/3/library/json.html)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)
- [NumPy argpartition](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [uv documentation](https://docs.astral.sh/uv/)
- [Sabofxx Call Me Maybe](https://github.com/Sabofxx/call-me-maybe): consulted
  for comparison of greedy constrained decoding and public SDK usage.
- [Kheldin Call Me Maybe](https://github.com/Kheldin/call-me-maybe): consulted
  for project structure and constrained decoding concepts.

AI assistance was used to review the subject and evaluation criteria, investigate
output and timing problems, refactor the model adapter and constraints, implement
validation and evaluation tooling, and draft regression tests and documentation.
The student must be able to explain and modify these components during defense.
