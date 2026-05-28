"""Build a chat/completions request file from a BFCL benchmark category.

Reads data/BFCL_v3_<category>.json, converts each entry's `function` specs to
OpenAI tool-calling format, and writes a JSON file shaped as a list-of-lists
of request objects.

The conversion logic mirrors bfcl.model_handler.utils.convert_to_tool with
ModelStyle.OpenAI. It is inlined here to avoid pulling in tree_sitter.
"""

import argparse
import jsonlines
import copy
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

GORILLA_TO_OPENAPI = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}


def _cast_to_openai_type(properties, mapping):
    for key, value in properties.items():
        if "type" not in value:
            properties[key]["type"] = "string"
        else:
            var_type = value["type"]
            if var_type == "float":
                properties[key]["format"] = "float"
                properties[key]["description"] = (
                    properties[key].get("description", "") + " This is a float type value."
                )
            properties[key]["type"] = mapping.get(var_type, "string")

        if properties[key]["type"] in ("array", "object"):
            if "properties" in properties[key]:
                properties[key]["properties"] = _cast_to_openai_type(
                    properties[key]["properties"], mapping
                )
            elif "items" in properties[key]:
                items = properties[key]["items"]
                if "type" in items:
                    items["type"] = mapping.get(items["type"], "string")
                if items.get("type") == "array" and "items" in items:
                    items["items"]["type"] = mapping.get(items["items"].get("type"), "string")
                elif items.get("type") == "object" and "properties" in items:
                    items["properties"] = _cast_to_openai_type(items["properties"], mapping)
    return properties


def convert_to_openai_tools(functions):
    """OpenAI-style equivalent of bfcl convert_to_tool(..., ModelStyle.OpenAI)."""
    tools = []
    for item in copy.deepcopy(functions):
        if "." in item["name"]:
            item["name"] = re.sub(r"\.", "_", item["name"])
        item["parameters"]["type"] = "object"
        item["parameters"]["properties"] = _cast_to_openai_type(
            item["parameters"]["properties"], GORILLA_TO_OPENAPI
        )
        tools.append({"type": "function", "function": item})
    return tools

CATEGORIES = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "java",
    "javascript",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
]


def build_requests(category: str, model: str, temperature: float):
    src = DATA_DIR / f"BFCL_v3_{category}.json"
    if not src.exists():
        raise FileNotFoundError(f"BFCL data file not found: {src}")

    requests = []
    with src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            messages = entry["question"][0]
            functions = entry.get("function", [])
            if not isinstance(functions, list):
                functions = [functions]
            tools = convert_to_openai_tools(functions)

            body = {"messages": messages}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "required"
            body.update(
                {
                    "temperature": temperature,
                    "stream": False,
                    "stream_options": {"include_usage": True},
                    "model": model,
                    "request_id": entry["id"]
                }
            )
            requests.append(body)
    return requests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", required=True, choices=CATEGORIES)
    p.add_argument("--model", required=True, help="Model name to embed in each request body")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--output",
        type=Path,
        default='bfcl_simple_required.jsonl',
        help="Output path (default: bfcl_requests_<category>_<model>.json next to this script)",
    )
    args = p.parse_args()

    requests = build_requests(args.category, args.model, args.temperature)

    with jsonlines.open(args.output, 'w') as f:
        for jsonobj in requests:
            f.write(jsonobj)

    print(f"Wrote {len(requests)} requests to {args.output}")


if __name__ == "__main__":
    main()
