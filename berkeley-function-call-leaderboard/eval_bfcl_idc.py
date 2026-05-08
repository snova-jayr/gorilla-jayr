import argparse
import json
import sys
import types
import yaml
from pathlib import Path

# --- Stub bfcl.constants.model_config so importing ast_checker doesn't pull in
# every model handler (anthropic, tree_sitter, vertexai, ...). The only field
# ast_checker reads is .underscore_to_dot, which we hardcode to True (matches
# OpenAI/Mistral/Google handlers — see model_config.py).
def _stub_bfcl_model_config():
    if "bfcl.constants.model_config" in sys.modules:
        return
    mod = types.ModuleType("bfcl.constants.model_config")

    class _StubConfig:
        underscore_to_dot = True

    class _StubMapping(dict):
        def __getitem__(self, key):
            return _StubConfig()
        def __contains__(self, key):
            return True

    mod.MODEL_CONFIG_MAPPING = _StubMapping()
    sys.modules["bfcl.constants.model_config"] = mod


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
_stub_bfcl_model_config()
from bfcl.eval_checker.ast_eval.ast_checker import ast_checker  # noqa: E402

DATA_DIR = REPO_ROOT / "data"


def decode_ast(tool_calls):
    """Inlined from bfcl OpenAIHandler.decode_ast (FC branch).

    Takes raw OpenAI tool_calls list, e.g.
        [{'id': '...', 'type': 'function',
          'function': {'name': 'math_factorial', 'arguments': '{"number": 5}'}}, ...]
    and returns BFCL-decoded form:
        [{'math_factorial': {'number': 5}}, ...]
    """
    # First, reshape to the BFCL "model_result" intermediate form: [{name: args_json_str}, ...]
    result = [{tc["function"]["name"]: tc["function"]["arguments"]} for tc in tool_calls]

    decoded_output = []
    for invoked_function in result:
        name = list(invoked_function.keys())[0]
        params = json.loads(invoked_function[name])
        decoded_output.append({name: params})
    return decoded_output

def read_predictions(resultslog: Path):
    preds = dict()
    msg_bodies = dict()
    categories = set()
    with open(resultslog) as f:
        for line in f:
            if 'Adding PEF' in line:
                pef_path = line.split('Adding PEF ')[-1].strip()[:-len(' with ID 0')]
                print(f'Reading results from pef path: {pef_path}')
            if ' loading checkpoint ' in line:
                ckpt_path = line.split(' loading checkpoint ')[-1].strip().split(', reuse_ckpt ')[0]
                print(f'Generated with ckpt path: {ckpt_path}')
            if '"name": "IDC_DEBUG"' in line and 'request_id' in line:
                request_response = yaml.safe_load(line)
                if 'Response: ' not in request_response['msg']:
                    continue
                msg_body_text = request_response['msg'].split('Response: ')[-1]
                msg_body_text = msg_body_text.replace("\\\\'}, 'logprobs'", "'}, 'logprobs'")
                msg_body_text = msg_body_text.replace("\\'", '<single-quote>')
                msg_body_text = msg_body_text.replace('\\<', '<').replace('\\>', '>').replace('\\|', '|')
                try:
                    msg_body = yaml.safe_load(msg_body_text.encode('utf-8', errors='replace').decode('utf-8'))
                except Exception as e:
                    print(f'Failed to parse response YAML ({e}); skipping. Raw text: {msg_body_text!r}')
                    continue
                bfcl_id = msg_body['body']['id']
                # bfcl_id looks like 'simple_0' or 'live_simple_0-0-0'. Strip the
                # trailing index segment to recover the category. The index segment
                # is the last underscore-delimited token, but it may itself contain
                # dashes (e.g. '0-0-0'), so we don't try to int() it.
                category = bfcl_id.rsplit('_', 1)[0]
                categories.add(category)
                message = msg_body['body']['choices'][0]['message']
                pred = message.get('tool_calls')
                if not pred:
                    # No tool_calls — fall back to content (e.g. model produced text
                    # instead of calling a function). evaluate() treats non-list values
                    # as "no function call decoded".
                    pred = message.get('content', '')
                if bfcl_id in preds:
                    print(f'Duplicate id {bfcl_id}; overwriting previous prediction.')
                preds[bfcl_id] = pred
                msg_bodies[bfcl_id] = msg_body # for debugging

    if len(categories) > 1:
        raise ValueError(f"Log mixes multiple BFCL categories: {sorted(categories)}")
    category = categories.pop() if categories else None
    return preds, msg_bodies, category

def _load_jsonl(path: Path):
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _language_for(category: str) -> str:
    if category == "java":
        return "Java"
    if category == "javascript":
        return "JavaScript"
    return "Python"


def evaluate(preds: dict, category: str, model_name: str = "gpt-4o-2024-11-20-FC"):
    """Score `preds` (bfcl_id -> raw OpenAI tool_calls list) against BFCL ground truth.

    Returns (accuracy, correct, total, errors).
    """
    prompt_entries = _load_jsonl(DATA_DIR / f"BFCL_v3_{category}.json")
    prompt_by_id = {e["id"]: e for e in prompt_entries}

    is_relevance = "relevance" in category or "irrelevance" in category
    pa_by_id = {}
    if not is_relevance:
        pa_entries = _load_jsonl(DATA_DIR / "possible_answer" / f"BFCL_v3_{category}.json")
        pa_by_id = {e["id"]: e for e in pa_entries}

    language = _language_for(category)

    correct = 0
    total = 0
    errors = []

    for bfcl_id, pred in preds.items():
        if bfcl_id not in prompt_by_id:
            print(f"Skipping {bfcl_id}: not found in {category} prompt file")
            continue

        decoded = None
        decode_err = None
        if isinstance(pred, list):
            try:
                decoded = decode_ast(pred)
            except Exception as e:
                decode_err = str(e)
        else:
            # Model returned content (string) instead of tool_calls. Treat as no
            # function call decoded — correct for irrelevance, fail for everything else.
            decoded = []
            decode_err = f"Model returned content instead of tool_calls: {pred!r}"

        if "irrelevance" in category:
            contain_func_call = decoded is not None and len(decoded) > 0
            valid = not contain_func_call
            err = ["Decoded a function call when none was expected."] if not valid else []
            err_type = "irrelevance_error:decoder_success"
        elif "relevance" in category:
            contain_func_call = decoded is not None and len(decoded) > 0
            valid = contain_func_call
            err = [f"Failed to decode a function call. {decode_err}"] if not valid else []
            err_type = "relevance_error:decoder_failed"
        else:
            if decoded is None:
                valid = False
                err = [f"Failed to decode AST. {decode_err}"]
                err_type = "ast_decoder:decoder_failed"
            else:
                verdict = ast_checker(
                    prompt_by_id[bfcl_id]["function"],
                    decoded,
                    pa_by_id[bfcl_id]["ground_truth"],
                    language,
                    category,
                    model_name,
                )
                valid = verdict["valid"]
                err = verdict.get("error", [])
                err_type = verdict.get("error_type", "")

        total += 1
        if valid:
            correct += 1
        else:
            errors.append({
                "id": bfcl_id,
                "error_type": err_type,
                "error": err,
                "decoded": decoded,
                "pred": pred,
            })

    accuracy = correct / total if total else 0.0
    return accuracy, correct, total, errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eventslog', type=str, required=True)
    parser.add_argument('--category', type=str, default=None,
                        help='BFCL category. If omitted, inferred from prediction ids in the log.')
    parser.add_argument('--model-name', type=str, default='gpt-4o-2024-11-20-FC',
                        help='Used by ast_checker for func-name normalization; any FC model works')
    parser.add_argument('--errors-out', type=Path, default=None,
                        help='Optional path to dump per-entry errors as JSON')
    args = parser.parse_args()

    preds, msg_bodies, log_category = read_predictions(Path(args.eventslog))
    print(f"Loaded {len(preds)} predictions")

    category = args.category or log_category
    if category is None:
        raise SystemExit("Could not infer category from log and --category was not provided")
    if args.category and log_category and args.category != log_category:
        print(f"Warning: --category={args.category!r} overrides category {log_category!r} found in log")

    accuracy, correct, total, errors = evaluate(preds, category, args.model_name)
    print(f"Category: {category}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Errors:   {len(errors)}")

    if args.errors_out is not None:
        with args.errors_out.open("w") as f:
            json.dump(errors, f, indent=2, default=str)
        print(f"Wrote per-entry errors to {args.errors_out}")