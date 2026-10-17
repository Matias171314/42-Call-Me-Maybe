*This project has been created as part of the 42 curriculum by mvasquez.*

# Call Me Maybe

## Description

Convert natural-language requests into structured function calls using
Qwen/Qwen3-0.6B and the supplied LLM SDK. The program chooses a function and
extracts its input arguments. It produces a description of the call; it does
not execute the function or return its calculated result.

The model decides what the request means. Constrained decoding controls which
continuations are allowed while that decision is generated. Successful outputs
have the required JSON structure and primitive argument types, although their
meaning can still be incorrect.

## Instructions

Python 3.10+ and `uv` are required. Keep the supplied `llm_sdk` directory next to
`src`. The first execution may download the model and needs network access.
Run the commands from the project root.

```sh
make install
make run
make help
```

| Command | Purpose |
| --- | --- |
| `make install` | Install dependencies with `uv sync` |
| `make run` | Execute the CLI with its default paths |
| `make debug` | Execute the CLI under Python's `pdb` debugger |
| `make lint` | Run flake8 and mypy with the required flags |
| `make lint-strict` | Run flake8 and strict mypy checks |
| `make clean` | Remove application bytecode and the project mypy cache |
| `make clean-venv` | Delete `.venv`; recreate it with `make install` |
| `make help` | Display the available commands |

Custom paths can be supplied directly:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

These are also the default paths. The Makefile forwards `ARGS` to both `run`
and `debug`:

```sh
make run ARGS="--functions_definition /tmp/functions.json --input /tmp/prompts.json --output /tmp/results.json"
make debug ARGS="--input /tmp/prompts.json --output /tmp/results.json"
```

Quote individual paths containing spaces inside `ARGS`:

```sh
make run ARGS='--input "/tmp/input files/prompts.json" --output "/tmp/output files/results.json"'
```

Missing output directories are created automatically. Invalid inputs and
processing failures produce a clear message and a nonzero exit code. Empty
prompt arrays produce `[]` without loading the LLM. Thread settings are left to
the runtime.

The lint commands exclude the virtual environment and the unchanged SDK.
Cleanup is scoped to application caches. Test programs and grading tools are
kept outside the submission repository.

## Input and output

Both input files contain JSON arrays. Each prompt entry requires a string field
`prompt`. Each function definition contains `name`, `description`, `parameters`
and `returns`; each parameter and return definition contains `type`.
Argument generation supports `string`, `number`, `integer` and `boolean`.

The output is an array whose entries contain exactly `prompt`, `name` and
`parameters`, following the supplied subject:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

The arguments are `2` and `3`; the program does not calculate `5`.

## Execution flow

1. `src/__main__.py` reads CLI options and parses the two input files.
2. `src/parsing/` uses Pydantic to validate records before model loading.
3. `Model` loads one SDK instance, reused for all requests.
4. `FunctionCallingEngine` describes the available functions and asks the model
   to select a name. The decoder limits the answer to those names or `NO_MATCH`.
   A `NO_MATCH` result raises an error; there is no forced second selection.
5. The engine requests each required argument separately, using its name, type,
   function description and original request. Already generated arguments form
   the partial JSON context for the next argument.
6. The decoder generates a value using the appropriate constraints. Python
   builds the keys and surrounding JSON structure, then `main` saves the results.

Pydantic validates the input records. The model adapter, function-calling engine
and constrained decoder are ordinary Python classes with constructors and
methods.

## How the LLM generates a continuation

The LLM operates on token IDs, not directly on Python strings. A token may
represent a word fragment, whitespace or punctuation; a word is not necessarily
a single token. The tokenizer assigns each token a numeric ID in its vocabulary.

For every generation step, `Model.get_logits()`:

1. Builds the Qwen conversation format using the instructions and partial answer.
   The wrapper supplies an empty thinking block before the answer continuation.
2. Calls the public SDK `encode()` method to convert that context into token IDs.
3. Calls `get_logits_from_input_ids()` to obtain one score for every possible
   next token. The array position is the token ID; its value is the token's logit.

A logit is a raw score, not a probability. Higher scores are preferred by the
model. Greedy selection needs only their ordering, so the program does not
convert them to probabilities or sample randomly.

The SDK loads pretrained model weights. Our code supplies context and selects
continuations; it does not train or modify those weights. After accepting a
continuation, the next model call includes it in the context. This repeats until
the value is complete.

`Model.decode_token()` calls the public SDK `decode()` method to recover a token's
text and caches the result. This avoids repeated decoding of the same ID. It
is not a cache of model predictions: the SDK still evaluates the complete input
context for each next-token call. Context length and the number of generation
steps therefore affect execution time.

## How constrained decoding works

### Selecting a valid token

`ConstrainedDecoder._select_valid_token()` copies the logit array and repeats:

```text
Take the token ID with the highest remaining score using argmax.
Decode its text and check the proposed continuation.
If it is nonempty and valid, accept it.
Otherwise set its score to negative infinity and try again.
```

Setting a rejected score to `-inf` prevents that token from winning the next
selection. The first accepted token is the highest-scoring valid continuation.
If none remains, the decoder raises a clear error.

For example, at the beginning of a numeric value, the following illustrative
scores could occur:

| Decoded token | Initial score | Decision |
| --- | --- | --- |
| `hello` | 9.5 | Invalid; mask to `-inf` |
| `7` | 8.2 | Valid; accept |
| `-` | 7.1 | Valid prefix, but lower-scoring |

The validator checks the accumulated text plus the candidate token. It does not
judge the token independently: `-` may start a number but cannot follow `12`.

### Function names, booleans and literal choices

`select_from_candidates()` only accepts prefixes that can reach an allowed
candidate. With `fn_add` and `fn_reverse`, `fn_a` remains possible while `fn_xyz`
is rejected. Shared prefixes are preserved: `foo` must not terminate immediately
when `foobar` is also a candidate unless an appropriate terminator is generated.
Once a nonempty prefix identifies a single candidate, its remaining suffix can
be completed without another model call.

The same mechanism selects booleans from `true` and `false`, and selects exact
literal strings from dynamically constructed candidates.

### Numbers

`generate_number()` checks numeric syntax step by step without regular
expressions. An unfinished prefix such as `-`, `3.` or `2e+` can still be extended,
but it cannot be returned as a completed number. Leading zeros such as `01` and
invalid transitions such as `1.e` are rejected.

When a delimiter is reached, `json.loads()` checks that the text is a complete
JSON number. Integer arguments reject decimal and exponent forms and are
converted directly to `int`, preserving large integer precision. Number
arguments become finite Python floats. NaN and infinity are not emitted.

### Strings and limits

`generate_string()` starts after an opening quote supplied by the engine. It
validates JSON escapes and stops at an unescaped closing quote. An escaped quote
belongs to the text; it does not close the string. `json.loads()` converts the
completed escaped content into a Python string.

Names and numbers have a default limit of 32 generated tokens; strings and
literal-copy selection use 128. An exhausted budget raises an error rather than
returning a truncated value. These are token limits, not character limits.

## Copying literals and deriving values

For each string argument, the model chooses between two representations:

- `copy`: select an exact fragment from the request. Candidate spans are built
  using general whitespace and quotation boundaries and encoded as JSON strings.
  This preserves characters such as embedded quotes and backslashes.
- `derive`: generate an input value implied by the request, using the same JSON
  string constraints. This is needed when the required value is not given
  literally, such as an expression or symbol described in words.

The code does not choose a representation from the parameter name. The model
makes that decision from the function definition and request. Copy constraints
ensure that the selected text exists in the input; they cannot guarantee that
the model chose the right fragment or representation.

## Hardcoding and generalization

Rules based on known example answers would make the program depend on those
examples. For instance, always returning a fixed pattern when a request contains
a particular category word, or filling parameters with memorized values for
known prompts, would bypass the model's interpretation. Such patches can raise
an example score while failing on changed wording, functions or argument roles.

The implementation therefore has no answer tables, function-specific extraction
branches, named-character replacement tables or post-generation text repairs.
Function selection uses the LLM. The same extraction process applies to all
parameter names, and literal candidates come from the current request.

JSON field names, boolean literals, number syntax and Qwen chat markers are
format rules, not stored solutions to test cases. Handling quoting, escapes and
numeric grammar correctly is also a general fix rather than memorizing answers.

Avoiding hardcoding does not make semantic mistakes acceptable or unavoidable.
Better general instructions and more reliable interpretation can improve the
results. The distinction is whether a change solves a general problem or merely
substitutes the expected answer for a known example.

## Public example results and known limitations

The measured result on the 11 public examples is **9/11 (81.8%)**, with all 11
outputs parseable and schema-compliant. The selected function names are correct;
the two failures concern argument meaning.

| Request | Observed error | Required interpretation |
| --- | --- | --- |
| Replace all numbers with `NUMBERS` | Uses `34 I'm 233` as the pattern | A pattern matching digit sequences, such as `\d+` |
| Replace all vowels with asterisks | Uses `asterisks` as the pattern and the source text as replacement | A vowel pattern and an asterisk replacement |

In the first case, the generated pattern matches one literal block instead of
each number. In the second, arguments are confused and the pattern does not
match the intended targets. Both calls remain valid JSON with string arguments;
their behavior is wrong. Equivalent correct patterns need not use identical
spelling, but they must express the requested operation.

Constrained decoding can reject malformed JSON and incorrect primitive types.
It cannot reject every semantically wrong string using those syntax rules alone.
The model can misclassify an implied value as literal text or select the wrong
fragment. The public semantic score remains below the 90% target; it is a known
limitation, not a claim of complete accuracy or an intentional objective.

The public run took **101.72 seconds**, including model loading, on CPU with a
cached model and normal runtime thread settings. This is a measured execution,
not a timing guarantee for other machines or uncached downloads.

Development checks pass 21 regression tests, flake8 and strict mypy. They cover
token rejection, shared prefixes, incomplete numbers, large integers, string
escapes, exact literal copying and input errors. Test programs and grading
artifacts stay outside the submission. The documented installation has also
been checked in a fresh virtual environment.

## Resources

- [Python JSON documentation](https://docs.python.org/3/library/json.html)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)
- [NumPy argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)

AI assistance was used to review requirements, compare implementations, refactor
code and run external checks. The student should understand and be able to
modify the decoding logic during the defense.
