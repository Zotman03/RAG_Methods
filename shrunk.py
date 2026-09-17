from __future__ import annotations

import argparse
import importlib
import json
import os
from functools import lru_cache
from typing import Any


SUMMARY_PROMPT = """Summarize the passage for retrieval and reranking.
Keep the summary concise but preserve the key facts, names, dates, numbers,
relationships, and technical terms. Do not add information that is not in the
passage. Return only the summary, without labels or commentary."""


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> tuple[Any, Any, Any]:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return torch, tokenizer, model


def summarize_chunk(
    chunk: str,
    model_name: str | None = None,
    max_new_tokens: int = 256,
) -> str:
    """Summarize one passage with the local Qwen model."""
    chunk = chunk.strip()
    if not chunk:
        raise ValueError("chunk must not be empty")

    try:
        importlib.import_module("torch")
        importlib.import_module("transformers")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Local inference requires torch and transformers. "
            "Run: python3 -m pip install torch transformers accelerate"
        ) from error

    model_name = model_name or os.environ.get("QWEN_MODEL", "Qwen/Qwen3-1.7B")
    torch, tokenizer, model = _load_model(model_name)
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": chunk},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    summary = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    if not summary:
        raise ValueError("model returned an empty summary")
    return summary


def create_chunk_pairs(
    chunks: list[str],
    model_name: str | None = None,
    max_new_tokens: int = 256,
) -> list[dict[str, str]]:
    """Return summary/original pairs for a collection of passage chunks."""
    pairs = []
    for chunk in chunks:
        original = chunk.strip()
        if not original:
            continue
        pairs.append(
            {
                "small_chunk": summarize_chunk(
                    original,
                    model_name=model_name,
                    max_new_tokens=max_new_tokens,
                ),
                "big_chunk": original,
            }
        )
    if not pairs:
        raise ValueError("at least one non-empty chunk is required")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize passage chunks and pair each summary with its original text."
    )
    parser.add_argument("text", nargs="*", help="One passage chunk")
    parser.add_argument(
        "--chunk",
        action="append",
        default=[],
        help="Passage chunk; repeat this option for multiple chunks",
    )
    parser.add_argument(
        "--file",
        help="Text file containing passages separated by blank lines",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-1.7B"),
        help="Local Hugging Face model name",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    chunks = []
    if args.text:
        chunks.append(" ".join(args.text))
    chunks.extend(args.chunk)
    if args.file:
        with open(args.file, encoding="utf-8") as passage_file:
            chunks.extend(passage_file.read().split("\n\n"))

    pairs = create_chunk_pairs(
        chunks,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(pairs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
