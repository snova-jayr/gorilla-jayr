import argparse
import json
import sys
import types
import yaml
import jsonlines
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

def read_predictions(output_jsonl: Path):
    tool_calls = []
    with jsonlines.open(output_jsonl) as f:
        for jsonobj in f:
            tool_calls.append(jsonobj['tool_calls'])
    return tool_calls

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


def evaluate(tool_calls: list[list[dict]], category: str, model_name: str = "gpt-4o-2024-11-20-FC"):
    """Score `tool_calls` (bfcl_id -> raw OpenAI tool_calls list) against BFCL ground truth.

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

    for idx, tc_list in enumerate(tool_calls):
        bfcl_id = f'{category}_{idx}'
        if bfcl_id not in prompt_by_id:
            print(f"Skipping {bfcl_id}: not found in {category} prompt file")
            continue

        if tc_list is None or len(tc_list) == 0:
            decoded = []
            decode_err = f"Model did not return valid tool_calls: {tc_list!r}"
        else:
            decoded = decode_ast(tc_list)
            decode_err = None

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
                "tool_calls": tc_list,
            })

    accuracy = correct / total if total else 0.0
    return accuracy, correct, total, errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-jsonl', type=str, required=True)
    parser.add_argument('--category', type=str, default=None,
                        help='BFCL category. If omitted, inferred from prediction ids in the log.')
    parser.add_argument('--model-name', type=str, default='gpt-4o-2024-11-20-FC',
                        help='Used by ast_checker for func-name normalization; any FC model works')
    parser.add_argument('--errors-out', type=Path, default=None,
                        help='Optional path to dump per-entry errors as JSON')
    args = parser.parse_args()

    tool_calls = read_predictions(Path(args.output_jsonl))
    print(f"Loaded {len(tool_calls)} predictions")

    category = args.category
    if category is None:
        raise SystemExit("Could not infer category from log and --category was not provided")

    accuracy, correct, total, errors = evaluate(tool_calls, category, args.model_name)
    print(f"Category: {category}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Errors:   {len(errors)}")

    if args.errors_out is not None:
        with args.errors_out.open("w") as f:
            json.dump(errors, f, indent=2, default=str)
        print(f"Wrote per-entry errors to {args.errors_out}")
