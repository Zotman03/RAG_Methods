from __future__ import annotations

import argparse
from functools import lru_cache
import importlib
import os
import sys
from typing import Any


SYSTEM_PROMPT = """Break the user's question into the smallest useful set of
independent subquestions for a retrieval-augmented generation system.

Return:
1. A numbered list of self-contained subquestions.
2. One short instruction explaining how to combine the answers.

Do not answer the question. If it is already a simple single-step question,
return it unchanged as the only subquestion.

Use only facts required by the original question. Never invent adjacent years,
extra time periods, entities, or comparisons that the user did not request.
Every subquestion must be independently searchable. Repeat the complete
subject, including the exact name of every person, place, organization, river,
or other entity from the original question. Never replace a named entity with
generic words such as "the river" or "the person".
For a question asking how a value changed from year A to year B, create exactly
two retrieval subquestions: the value in year A and the value in year B.
Then state that those two values should be subtracted to calculate the change.

Example:
Question: What is the average ocean temperature rise from 1990-2000?
1. What was the average ocean temperature in 1990?
2. What was the average ocean temperature in 2000?
Combine the two retrieved values to calculate the change from 1990 to 2000.

Example:
Question: How fast did the Yangtze River flow from 1800 to 1910?
1. What was the flow rate of the Yangtze River in 1800?
2. What was the flow rate of the Yangtze River in 1910?
Combine the two retrieved Yangtze River flow rates to calculate the change
from 1800 to 1910.

Do not create a question about 2000-2010 or any other period unless it appears
in the user's question. Before responding, check that every subquestion still
contains the original named entities and the requested years."""


@lru_cache(maxsize=2)
def _load_model(model_name: str, device: str) -> tuple[Any, Any, Any]:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    dtype = torch.float16 if device == "mps" else "auto"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    device_map = {"": device} if device == "mps" else "auto"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    return torch, tokenizer, model


def break_question(
    question: str,
    model_name: str | None = None,
    max_new_tokens: int = 256,
    device: str | None = None,
) -> str:
    """Run Qwen locally and return its question breakdown."""
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    try:
        importlib.import_module("torch")
        importlib.import_module("transformers")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Local inference requires torch and transformers. "
            "Run: python3 -m pip install torch transformers accelerate"
        ) from error

    model_name = model_name or os.environ.get("QWEN_MODEL", "Qwen/Qwen3-1.7B")
    torch = importlib.import_module("torch")
    device = device or os.environ.get("QWEN_DEVICE", "auto")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; use --device cpu instead")

    load_description = "CPU/disk offload" if device == "auto" else device
    print(f"Loading {model_name} with {load_description}...", file=sys.stderr, flush=True)
    torch, tokenizer, model = _load_model(model_name, device)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Generating breakdown...", file=sys.stderr, flush=True)
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Break a question into retrieval-friendly subquestions locally."
    )
    parser.add_argument("question", nargs="+", help="Question to break down")
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-1.7B"),
        help="Local Hugging Face model name",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--device",
        default=os.environ.get("QWEN_DEVICE", "auto"),
        choices=["auto", "mps", "cpu"],
        help="Inference device; auto uses Accelerate CPU/disk offload",
    )
    args = parser.parse_args()
    print(
        break_question(
            " ".join(args.question),
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
