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
Cleanup is scoped to application caches.

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
   function description and original request. Literal extraction uses already
   generated arguments as partial JSON context; derived strings use a JSON
   object containing only the current argument.
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

For each string argument, the model first identifies its representation:

- `literal`: select an exact fragment from the request. Candidate spans are built
  using general whitespace and quotation boundaries and encoded as JSON strings.
  This preserves characters such as embedded quotes and backslashes.
- `expression`: generate a regular expression describing the requested matching
  rule, rather than listing matching text from the example.
- `symbol`: generate the actual character described by its name.

This decision has its own prompt, separate from the instruction to preserve
literal text. A matching rule may require notation that does not appear anywhere
in the request; restricting it to copied fragments would prevent the correct
value from being generated. The representation label is an internal decision
and is not included in the output JSON.

Expressions and symbols are generated in a JSON object containing only the
current argument. The decoder still enforces valid JSON string syntax. Literal
extraction instead uses the partial arguments object and preserves the exact
selected input text.

Literal extraction includes one further model choice between the extracted text,
the characters in Python's `string.punctuation`, and a space. This lets the model
interpret named punctuation even when it initially classified the value as
literal text. The second choice cannot replace a name with another fragment of
the request. There is no dictionary translating character names to symbols.

The code does not choose a representation from the parameter name. The model
makes that decision from the function definition and request. Literal constraints
preserve the chosen fragment, and the punctuation choice limits alternatives to
individual characters. Neither constraint guarantees the correct meaning.

## Hardcoding and generalization

Rules based on known example answers would make the program depend on those
examples. For instance, always returning a fixed pattern when a request contains
a particular category word, or filling parameters with memorized values for
known prompts, would bypass the model's interpretation. Such patches can raise
an example score while failing on changed wording, functions or argument roles.

The implementation therefore has no answer tables, function-specific extraction
branches, named-character replacement tables or deterministic answer repairs.
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

The measured result on the 11 examples in `data/input` is **11/11 (100%)**.
All outputs are parseable and schema-compliant. All selected function names
are correct, and the generated arguments produce the expected public example
results.

| Request | Generated pattern | Generated replacement |
| --- | --- | --- |
| Replace all numbers with `NUMBERS` | `([0-9]+)` | `NUMBERS` |
| Replace all vowels with asterisks | `([aeiouAEIOU])` | `*` |

These patterns are generated by the model, not stored in the engine. Equivalent
correct patterns need not use identical spelling, but they must express the
requested operation.

Constrained decoding can reject malformed JSON and incorrect primitive types.
It cannot reject every semantically wrong string using those syntax rules alone.
The model can misclassify an implied value as literal text or select the wrong
fragment or punctuation character. Passing these 11 examples does not guarantee
correct interpretation of new requests or function definitions.

The public run took **127.64 seconds**, including model loading, on CPU with a
cached model and normal runtime thread settings. This is a measured execution,
not a timing guarantee for other machines or uncached downloads.

## Resources

- [Python JSON documentation](https://docs.python.org/3/library/json.html)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)
- [NumPy argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)

AI assistance was used to review requirements, compare implementations, refactor
code and check the public examples in `data/input`.
