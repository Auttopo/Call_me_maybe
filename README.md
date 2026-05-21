*This project has been created as part of the 42 curriculum by abenabde.*

---

# call me maybe — Introduction to Function Calling in LLMs

## Description

**call me maybe** is a project exploring **function calling** in Large Language Models (LLMs) through **constrained decoding** — a technique that guides a model's token-by-token generation to always produce syntactically valid and semantically meaningful output.

The goal is to take a set of function definitions (name, parameters, types) and a list of natural-language prompts, then have a small LLM (`Qwen/Qwen3-0.6B`) select the right function and fill in its arguments — all without relying on post-processing heuristics or retries. The output is a structured JSON array, where each entry maps an input prompt to a function call with properly typed parameters.

Instead of letting the model generate freely and hoping it produces valid JSON, this project **constrains the generation at every token step**: only tokens that could legally extend the current partial output are assigned non-`-inf` logits, making malformed output structurally impossible.

---

## Algorithm Explanation

### Constrained Decoding

At each generation step, the model produces a logit distribution over its vocabulary. Normally, you would take `argmax` or sample from it. Here, before picking a token, a mask is applied so that only "legal" tokens at the current position have finite logits — all others are set to `-inf`.

The generation follows a **state machine** defined by `create_sequence`, which encodes the expected output as a list of literal string segments interleaved with special markers:

| Marker | Meaning |
|--------|---------|
| String `"..."` | Literal text that must be generated verbatim |
| `-1` | Constrained function name slot (only valid function names allowed) |
| `-2` | Constrained parameter values slot |
| `-3` | End of sequence |

The overall JSON structure enforced per prompt is:

```
{"prompt": "<prompt>", "name": <function_name>, "parameters": {<key>: <value>, ...}}
```

### Function Name Selection (`contrain_func`)

When the state machine reaches a `-1` slot, only tokens that are valid **prefixes of known function names** are unmasked. Generation continues until the accumulated string exactly matches one of the known function names, at which point the parameter sub-sequence is created.

### Parameter Value Selection (`contrain_param` / `choose_param`)

Parameter values are typed (string, number, boolean, null). The `validate_case` function enforces type-correct generation:

- Strings must begin and end with `"`.
- Numbers must parse as valid floats.
- Booleans are matched against `true` / `false`.
- `null` is matched literally.

If a partially generated value cannot lead to a valid token of the right type, the token is rejected and the next best logit is tried.

### Strict vs. Non-Strict Mode

In **strict mode** (`--strict`), logits are re-fetched from the model at every single token step — maximally accurate but slower.

In **non-strict mode** (default), logits are only re-fetched when the generation is at a constrained decision point (function name or parameter value). For literal string segments that must be produced verbatim, the longest matching token is greedily selected from the existing logit vector without a new forward pass, saving significant inference time.

---

## Design Decisions

**Vocabulary pre-computation:** The full vocabulary is decoded once at startup into `dict_to_asc` (index → string) and `dict_to_log` (string → index). Duplicate token strings are tracked in a `black_list` so ambiguous tokens are never selected.

**Longest-token preference (`contrain_longer`):** When emitting literal text, the decoder always picks the longest vocabulary token that matches the remaining prefix. This minimizes the number of forward passes needed, which is the main bottleneck.

**Sub-sequence design:** Each parameter slot in the sequence is encoded as `[cursor, key1_literal, -1, key2_literal, -1, ..., -2, ""]`. The cursor (index 0) tracks where generation currently is, and `-1`/`-2` markers distinguish "still typing the value" from "finished the value".

**Pydantic for all I/O validation:** The `Parser` and associated models (`FuncDefsDict`, `InputsDict`, etc.) use Pydantic v2 with `extra="forbid"` to catch malformed input files immediately, with clear error messages.

**Output file safety:** `ParserOutput` automatically appends `_1`, `_2`, etc. to the output filename if a file already exists, preventing accidental overwrites.

---

## Instructions

### Requirements

- Python 3.13.*
- The `llm_sdk` library (provides `Small_LLM_Model`)
- `numpy`
- `tqdm`
- `pydantic`

Install dependencies:

```bash
make install
```

### Input File Format

**Function definitions** (`data/input/functions_definition.json`):

```json
[
  {
    "name": "get_weather",
    "description": "Returns the weather for a given city.",
    "parameters": {
      "city": { "type": "string" },
      "units": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]
```

**Prompts** (`data/input/function_calling_tests.json`):

```json
[
  { "prompt": "What is the weather in Paris in Celsius?" },
  { "prompt": "Convert 100 USD to EUR." }
]
```

### Running the Program

**Default paths** (uses `data/input/functions_definition.json`, `data/input/function_calling_tests.json`, outputs to `data/output/output.json`):

```bash
make run
```

**Custom paths:**

```bash
python main.py \
  --functions_definition path/to/functions.json \
  --input path/to/prompts.json \
  --output path/to/result.json
```

**Strict mode** (re-fetches logits at every token — slower but more accurate):

```bash
python main.py --strict
```

### Example Usage

Given a function definition for `send_email(to: string, subject: string, body: string)` and the prompt `"Send an email to alice@example.com about the meeting tomorrow"`, the output would be:

```json
[
  {
    "prompt": "Send an email to alice@example.com about the meeting tomorrow",
    "name": "send_email",
    "parameters": {
      "to": "alice@example.com",
      "subject": "Meeting tomorrow",
      "body": "Just a reminder about our meeting tomorrow."
    }
  }
]
```

---

## Performance Analysis

**Accuracy** is high for well-formed prompts that map clearly to one function. The constrained decoding guarantees that the output is always valid JSON with correctly typed parameters — no post-processing or retry logic needed. Ambiguity arises when a prompt could map to multiple functions or when parameter values are not explicit in the prompt text.

**Speed** depends on the number of model forward passes. In non-strict mode, forward passes are only triggered at genuine decision points (function name selection and parameter value generation), making inference substantially faster than strict mode. Literal segments of the output (brackets, keys, colons) are filled deterministically with no new forward pass.

**Reliability** is guaranteed for output structure: the state machine makes it impossible to produce malformed JSON. Type correctness for parameter values is enforced token by token.

---

## Challenges Faced

**Vocabulary ambiguity:** Some LLM tokenizers assign different indices to tokens with identical string representations. This required building the `black_list` to exclude duplicates and ensure `dict_to_log` maps each string to exactly one canonical index.

**Typed parameter generation:** Knowing when a string value is "complete" (the closing `"` has been emitted) required careful state tracking. The `validate_case` function handles the four distinct states a partially generated value can be in, including edge cases like strings that contain quotes from the tokenizer.

**Non-strict speed vs. correctness trade-off:** In non-strict mode, reusing stale logits for literal segments works because these segments are fully determined — but care was needed to ensure no constrained decision is made on stale logits.

**Longest-token selection:** Greedily picking the longest matching token rather than `argmax` required a separate pass over the vocabulary whitelist (`contrain_longer`) and is critical for performance.

---

## Testing Strategy

- **Unit testing of `validate_case`:** Edge cases for string (open, closed, containing internal quotes), number (int, float, negative, partial), boolean, and null values were tested individually.
- **End-to-end testing:** Known prompts with known expected function calls were run and the JSON output was compared against expected results.
- **Malformed input testing:** Invalid JSON, missing fields, and extra fields in input files were tested to verify Pydantic validation raises clear errors.
- **Strict vs. non-strict comparison:** Both modes were run on the same inputs to verify they produce identical outputs, confirming that non-strict mode's logit reuse does not degrade correctness for constrained segments.

---

## Bonus

### Performance Optimization — Non-Strict Mode (Logit Reuse)

The most significant optimization is the **non-strict generation mode** (active by default). A model forward pass is expensive; the key insight is that for literal segments of the output — fixed strings like `{"prompt": "`, `", "name": `, `", "parameters": {`, etc. — there is no decision to make. The tokens are fully determined by the expected output structure, so there is no need to query the model.

In non-strict mode, the code distinguishes between two situations at each step:

- **Constrained decision point** (function name or parameter value): the model is queried for fresh logits, and the whitelist mask is applied before selecting a token.
- **Literal segment**: `contrain_longer` selects the longest vocabulary token that matches the remaining literal prefix, using the **existing logit vector** without a new forward pass. After the token is appended, `logits[masque] = float("+inf")` resets the vector so the next literal token is also selected greedily without model involvement.

This means the model is only called when it genuinely needs to make a semantic choice, cutting the number of forward passes down to roughly `2 × number_of_parameters` per prompt instead of one per output token.

**Tokenizer batching:** All prompts are encoded in a single batch before generation begins, and all generated token sequences are decoded in a single batch at the end — avoiding repeated tokenizer calls inside the generation loop. Model inference itself remains sequential per prompt, as constrained decoding requires per-step logit masking that depends on the partially generated output of each individual sequence.

### Encoding and Decoding Integration with Constrained Decoding

The vocabulary pre-computation at startup is what makes token-level constraint enforcement practical. Rather than calling `encode`/`decode` on every step, the entire vocabulary is decoded once:

```python
dict_to_asc: list[str] = list(obj.decode([[i] for i in range(len(logits))]))
dict_to_log: dict[str, int] = {}
black_list = np.zeros(len(logits), dtype=bool)
for i, char in enumerate(dict_to_asc):
    if dict_to_log.get(char) is None:
        dict_to_log[char] = i
    else:
        black_list[i] = True  # duplicate string representation — never select
```

This produces two structures used throughout constrained decoding:

- `dict_to_asc` maps every token index to its string — used to check whether appending a token would still match the expected prefix (`contrain_basic`, `contrain_longer`).
- `dict_to_log` maps every string to its canonical token index — used to look up which index to unmask when building the whitelist in `contrain_func` and `contrain_longer`.
- `black_list` permanently masks tokens whose string representation is a duplicate of another token, eliminating ambiguity.

Encoding feeds into constrained decoding via `keeps[i]`, the running list of token IDs for each prompt. Every selected token index is appended to `keeps[i]`, which is passed back to `get_logits_from_input_ids` on the next constrained step to obtain the model's next distribution conditioned on all tokens generated so far.

### Visualization of the Generation Process

The program produces a simple graphical step-by-step visualization of the generation loop.

---

## Resources

### Documentation & References

- [A Guide to Structured Generation Using Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)
- [Part 6: Implementing Constrained Decoding](https://medium.com/@albersj66/part-6-implementing-constrained-decoding-for-phi-3-vision-2c72a1be6a17)
- local package file : `llm_sdk/llm_sdk/__init__.py`

### AI Usage

AI (Claude, Anthropic) was used in this project for the following tasks:

- **README writing:** Drafting and structuring this document based on the project.
- **LLM description:** Explaining how LLMs work, as well as common APIs and associated techniques.

The core algorithm design, constrained decoding logic, and all implementation were done by the project authors abenabde.
