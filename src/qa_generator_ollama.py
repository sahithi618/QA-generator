
from ast import Return
import csv
from http import client
import json
import re
import subprocess
import sys
import time
import requests
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)



# ── STOPWORDS (excluded from token overlap check) ────────────────────────────
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for",
    "on", "is", "are", "as", "with", "by", "that", "this",
    "it", "from", "at", "be", "our", "its", "their", "which",
    "we", "has", "have", "been", "were", "was", "will", "would",
    "not", "no", "but", "if", "than", "then", "so", "also"
}

# ── TYPE NORMALIZATION ────────────────────────────────────────────────────────
TYPE_MAP = {
    "fact":        "fact extraction",
    "numeric":     "numeric calculation",
    "comparison":  "comparison",
    "multi":       "multi-step reasoning",
    "reasoning":   "multi-step reasoning",
    "calculation": "numeric calculation",
}

def normalize_type(raw: str) -> str:
    raw = raw.lower().strip()
    for key, val in TYPE_MAP.items():
        if key in raw:
            return val
    return "fact extraction"

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


# ── STAGE 1: GENERATION ───────────────────────────────────────────────────────

def parse_jsonlines(text: str) -> list[dict]:
    """Extracts JSON objects from a response that may have extra text around them."""
    results = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        start = line.find("{")
        end   = line.rfind("}")
        if start == -1 or end == -1:
            continue
        try:
            obj = json.loads(line[start:end+1])
            results.append(obj)
        except json.JSONDecodeError:
            pass
    return results

GENERATION_PROMPT = """
Passage:
{text}

Generate up to 2 high-quality QA pairs, quality matters over quantity.

Requirements:
- Answer must be supported by the passage.
- Include source sentence.
- Assign:
  type = fact extraction | numeric calculation | comparison | multi-step reasoning
  difficulty = easy | medium | hard

Return JSON lines only.

{{"question":"","answer":"","source":"","type":"","difficulty":""}}

Return SKIP if no useful question exists.
"""

def generate_qa_groq(
    chunk_text: str,
    section: str = ""
) -> list[dict]:

    prompt = GENERATION_PROMPT.format(
        text=chunk_text
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1200
        )

        raw = response.choices[0].message.content

        if "SKIP" in raw:
            return []

        return parse_jsonlines(raw)

    except Exception as e:
        print(e)
        return []


# ── STAGE 2: VERIFICATION ─────────────────────────────────────────────────────

def _token_overlap(a: str, b: str) -> float:
    """Fraction of meaningful tokens in `a` that appear in `b`."""
    tokens_a = [t for t in re.findall(r"\w+", normalize_text(a)) if t not in STOPWORDS]
    tokens_b = set(re.findall(r"\w+", normalize_text(b)))
    if not tokens_a:
        return 0.0
    return sum(1 for t in tokens_a if t in tokens_b) / len(tokens_a)


def check_source_grounded(source: str, chunk_text: str) -> tuple[bool, str]:
    """
    Check A: Verify the source passage exists in the chunk.
    Accepts either exact substring match OR high token overlap (≥ 0.85).
    High threshold because source should be a near-verbatim quote.
    """
    if not source:
        return False, "source is empty"

    norm_source = normalize_text(source)
    norm_chunk  = normalize_text(chunk_text)

    # Exact substring match
    if norm_source in norm_chunk:
        return True, ""

    # Near-verbatim match (handles minor whitespace/punctuation differences)
    overlap = _token_overlap(source, chunk_text)
    if overlap >= 0.85:
        return True, ""

    return False, f"source not found in chunk (overlap={overlap:.2f})"


def is_clone(question, answer, source):

    norm_q = normalize_text(question)
    norm_a = normalize_text(answer)
    norm_s = normalize_text(source)

    if norm_a == norm_s:
        return True

    if norm_q in norm_a:
        return True

    return False

QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "which",
    "when",
    "who"
}

def is_specific_question(
    question
):

    q = question.lower()

    if len(q.split()) < 5:
        return False

    if not any(
        word in q
        for word in QUESTION_WORDS
    ):
        return False

    return True



def verify_qa_pair(qa, chunk_text):

    question = qa.get("question","")
    answer = qa.get("answer","")
    source = qa.get("source","")

    if is_clone(question, answer, source):
        return False, "[Clone]"

    if not is_specific_question(question):
        return False, "[Question Quality]"

    passed, reason = check_source_grounded(
        source,
        chunk_text
    )

    if not passed:
        return False, f"[Grounding] {reason}"

    return True, ""


def verify_dataset(raw_dataset: list[dict],
                   chunk_lookup: dict[int, str]) -> tuple[list[dict], list[dict]]:
    """
    Stage 2: Run verification across the entire generated dataset.
    
    chunk_lookup: {chunk_id -> chunk_text} built once before this call.
    Returns (verified_pairs, rejected_pairs).
    """
    verified = []
    rejected = []

    for qa in raw_dataset:
        chunk_id   = qa.get("chunk_id")
        chunk_text = chunk_lookup.get(chunk_id, "")

        passed, reason = verify_qa_pair(qa, chunk_text)
        if passed:
            verified.append(qa)
        else:
            rejected.append({**qa, "rejection_reason": reason})

    return verified, rejected



# ── STAGE 3: DEDUPLICATION ────────────────────────────────────────────────────

def dedupe_qa_pairs(qa_pairs: list[dict]) -> list[dict]:
    """Remove duplicate QA pairs by (question, answer) key."""
    seen   = set()
    unique = []
    for qa in qa_pairs:
        key = (
            normalize_text(qa.get("question", "")),
            normalize_text(qa.get("answer", ""))
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(qa)
    return unique


# ── FULL PIPELINE ─────────────────────────────────────────────────────────────


def run_pipeline(chunks: list[dict],
                 model: str,
                 num_questions: int = 2) -> tuple[list[dict], list[dict]]:
    """
    Runs the full pipeline:
      Stage 1 — Generate QA pairs for every chunk
      Stage 2 — Verify all pairs in a separate pass
      Stage 3 — Deduplicate
    """

    # Build chunk lookup once for the verification stage
    chunk_lookup = {
        chunk.get("chunk_id", i): chunk.get("text", "")
        for i, chunk in enumerate(chunks)
    }

    # ── Stage 1: Generation ───────────────────────────────────────────────────
    print("\n── Stage 1: Generating QA pairs ────────────────────────")
    raw_dataset = []
    failed_generation = []

    for idx, chunk in enumerate(chunks):
        chunk_id   = chunk.get("chunk_id", idx)
        chunk_text = chunk.get("text", "").strip()

        if not chunk_text:
            continue

        print(f"  [{idx+1}/{len(chunks)}] chunk {chunk_id}...", end=" ", flush=True)
        pairs = generate_qa_groq(chunk_text=chunk_text, section=chunk.get("section", "")
)

        if pairs:
            print(f"✓ {len(pairs)} pairs")
            for qa in pairs:
                raw_dataset.append({
                    "chunk_id":   chunk_id,
                    "filing":     chunk.get("filing", ""),
                    "section":    chunk.get("section", ""),
                    "item":       chunk.get("item", ""),
                    "question":   qa.get("question", "").strip(),
                    "answer":     qa.get("answer", "").strip(),
                    "source":     qa.get("source", "").strip(),
                    "type":       normalize_type(qa.get("type", "")),
                    "difficulty": qa.get("difficulty", "medium").strip(),
                })
        else:
            print("✗ no pairs generated")
            failed_generation.append(chunk_id)

    print(f"\n  Generated: {len(raw_dataset)} raw pairs")
    print(f"  Failed generation: {len(failed_generation)} chunks")

    # ── Stage 2: Verification ─────────────────────────────────────────────────
    print("\n── Stage 2: Verifying QA pairs ─────────────────────────")
    verified, rejected = verify_dataset(raw_dataset, chunk_lookup)

    
    print(f"  Passed : {len(verified)}")
    print(f"  Rejected: {len(rejected)}")

    if rejected:
        print("\n  Rejection breakdown:")
        reason_counts: dict[str, int] = {}
        for r in rejected:
            key = r["rejection_reason"].split("]")[0] + "]"
            reason_counts[key] = reason_counts.get(key, 0) + 1
        for reason, count in reason_counts.items():
            print(f"    {reason}: {count}")

    # ── Stage 3: Deduplication ────────────────────────────────────────────────
    print("\n── Stage 3: Deduplicating ───────────────────────────────")
    before = len(verified)
    verified = dedupe_qa_pairs(verified)
    print(f"  {before} → {len(verified)} after deduplication")

    return verified, rejected

    

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="10-K QA Generator with Verification")
    parser.add_argument("--setup",         action="store_true",   help="Run Ollama setup")
    parser.add_argument("--input",         default="chunks_10k.json")
    parser.add_argument("--output",        default="qa_10k_3.json")
    parser.add_argument("--rejected",      default="qa_rejected_3.json", help="Save rejected pairs here")
    parser.add_argument("--csv",           action="store_true",   help="Also save CSV")
    parser.add_argument("--num-questions", type=int, default=2)
    parser.add_argument("--limit",         type=int, default=None)
    parser.add_argument("--model",         default="phi3:mini")
    args = parser.parse_args()


    # Load chunks
    print(f"\nLoading chunks from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if args.limit:
        chunks = chunks[:args.limit]

    filing_name = Path(args.input).stem
    for chunk in chunks:
        chunk["filing"] = filing_name

    print(f"Loaded {len(chunks)} chunks | filing: {filing_name}")

    # Run pipeline
    verified, rejected = run_pipeline(chunks, args.model, args.num_questions)

    # Save verified
    print(f"\nSaving {len(verified)} verified QA pairs → {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(verified, f, indent=2, ensure_ascii=False)

    # Save rejected (useful for debugging prompt quality)
    print(f"Saving {len(rejected)} rejected pairs → {args.rejected}")
    with open(args.rejected, "w", encoding="utf-8") as f:
        json.dump(rejected, f, indent=2, ensure_ascii=False)

    # CSV output
    if args.csv:
        csv_path = Path(args.output).with_suffix(".csv")
        print(f"Saving CSV → {csv_path}")
        fields = ["chunk_id", "filing", "section", "item",
                  "question", "answer", "source", "type", "difficulty"]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(verified)

    # Summary
    print("\n" + "=" * 60)
    print(f"✓ Done")
    print(f"  Verified QA pairs : {len(verified)}")
    print(f"  Rejected pairs    : {len(rejected)}")
    print(f"  Output            : {args.output}")
    print(f"  Rejected log      : {args.rejected}")

    if verified:
        s = verified[0]
        print(f"\n  Sample:")
        print(f"    Q : {s['question'][:100]}")
        print(f"    A : {s['answer'][:100]}")
        print(f"    src: {s['source'][:80]}")


if __name__ == "__main__":
    main()