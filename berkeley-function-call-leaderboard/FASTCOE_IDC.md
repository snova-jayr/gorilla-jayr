# BFCL on FASTCoE/IDC

Two helper scripts at the repo root:

- [construct_bfcl_idc.py](construct_bfcl_idc.py) — turns a BFCL benchmark category into a chat-completions request file.
- [eval_bfcl_idc.py](eval_bfcl_idc.py) — parses an IDC events log of responses and scores them against BFCL ground truth.

Both scripts are self-contained: the relevant pieces of `bfcl.model_handler.utils.convert_to_tool` and `OpenAIHandler.decode_ast` are inlined so neither requires installing the `bfcl` package. The eval script does `import bfcl.eval_checker.ast_eval.ast_checker`, but stubs `bfcl.constants.model_config` to skip the heavy handler imports.

## 1. Construct requests

```bash
python construct_bfcl_idc.py \
    --category simple \
    --model gpt-oss-120b-8k \
    --output <your fast-coe repo root>/server/rdu_manifest/configs/acb_tp16_gpt_oss_bfcl.json
```

### Arguments

| Arg | Required | Default | Notes |
|---|---|---|---|
| `--category` | yes | — | One of the BFCL v3 categories (see below). |
| `--model` | yes | — | Model name embedded into every request body. |
| `--temperature` | no | `0.0` | |
| `--output` | no | `acb_tp16_gpt_oss_bfcl.json` next to the script | Output file path. |

### Categories

Multi-turn is intentionally excluded.

```
simple, multiple, parallel, parallel_multiple,
live_simple, live_multiple, live_parallel, live_parallel_multiple
```

### Output

A JSON file containing a list of request batches (list-of-lists). Each entry corresponds to one BFCL example:

Notes:
- `request_id` is the BFCL `id` (e.g. `simple_0`, `live_simple_0-0-0`). The eval script uses this to look up ground truth.

## 2. Evaluate predictions

```bash
python eval_bfcl_idc.py \
    --eventslog /path/to/idc_events.log
```

The script parses `IDC_DEBUG`-tagged lines from the log, extracts the response body for each request, and computes accuracy against BFCL ground truth.

### Arguments

| Arg | Required | Default | Notes |
|---|---|---|---|
| `--eventslog` | yes | — | Path to the IDC events log file. |
| `--category` | no | inferred from log | BFCL category. Inferred from the prediction ids in the log (e.g. `live_simple_0-0-0` → `live_simple`). If the log mixes categories, the script errors out and you must pick one explicitly. |
| `--errors-out` | no | — | Optional path to dump per-entry failures as JSON (`id`, `error_type`, `error`, `decoded`, raw `pred`). |

### Output

```
Loaded 400 predictions
Category: simple
Accuracy: 0.8525 (341/400)
Errors:   59
```

With `--errors-out errors.json`, each failing entry is recorded with the BFCL id, the checker's error message and category, the decoded model output (if any), and the raw prediction.

## End-to-end flow

```bash
# 1. Build requests
python construct_bfcl_idc.py --category simple --model gpt-oss-120b-8k --output /tmp/simple_requests.json

# 2. Run those requests on IDC (out of scope here) and capture the events log.

# 3. Score
python eval_bfcl_idc.py --eventslog /path/to/idc.log --errors-out /tmp/simple_errors.json
```

## Limitations

- Multi-turn categories (`multi_turn_*`) are not supported by either script.
- The eval script assumes one category per log; mixed-category logs need separate runs.
- `tool_choice` defaults to `"auto"`; edit the construct script if needed.
