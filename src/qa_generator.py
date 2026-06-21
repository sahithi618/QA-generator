"""
QA Pair Generator
Generates question-answer pairs for each chunk using an LLM.
Supports: Ollama (local, free), Hugging Face, or OpenAI API.
"""

import json
import sys
import os
from typing import Optional

# Try to import requests for API calls
try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system("pip install requests > nul 2>&1")
    import requests


def generate_qa_ollama(chunk_text: str, model: str = "llama2", num_questions: int = 3) -> list[dict]:
    """
    Generate QA pairs using Ollama (local LLM).
    Requires Ollama running on localhost:11434
    Download Ollama from https://ollama.ai
    """
    prompt = f"""Based on the following text, generate {num_questions} unique question-answer pairs.
Format each pair as JSON on a single line like this:
{{"question": "Question text?", "answer": "Answer text."}}

Text:
{chunk_text[:2000]}

Generate {num_questions} QA pairs (one JSON object per line):"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        text = result.get("response", "")
        
        qa_pairs = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if line and line.startswith('{'):
                try:
                    pair = json.loads(line)
                    if "question" in pair and "answer" in pair:
                        qa_pairs.append(pair)
                except json.JSONDecodeError:
                    pass
        return qa_pairs
    except Exception as e:
        print(f"  Error with Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return []


def generate_qa_huggingface(chunk_text: str, num_questions: int = 3) -> list[dict]:
    """
    Generate QA pairs using Hugging Face Inference API (free tier available).
    """
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("  HUGGINGFACE_API_KEY not set. Get free key from https://huggingface.co/settings/tokens")
        return []

    prompt = f"""Based on this text, generate {num_questions} question-answer pairs.
Return ONLY valid JSON lines (one per question).
Each line: {{"question": "...", "answer": "..."}}

Text: {chunk_text[:1000]}

JSON output:"""

    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(
            "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
            headers=headers,
            json={"inputs": prompt},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        text = result[0].get("generated_text", "")
        
        qa_pairs = []
        for line in text.split('\n'):
            line = line.strip()
            if line and line.startswith('{'):
                try:
                    pair = json.loads(line)
                    if "question" in pair and "answer" in pair:
                        qa_pairs.append(pair)
                except json.JSONDecodeError:
                    pass
        return qa_pairs
    except Exception as e:
        print(f"  Error with Hugging Face: {e}")
        return []


def generate_qa_openai(chunk_text: str, num_questions: int = 3) -> list[dict]:
    """
    Generate QA pairs using OpenAI API (paid, but low cost).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  OPENAI_API_KEY not set.")
        return []

    try:
        import openai
        openai.api_key = api_key
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"Generate {num_questions} question-answer pairs from the text. Return ONLY valid JSON lines."
                },
                {
                    "role": "user",
                    "content": f"""Text:
{chunk_text[:2000]}

Generate {num_questions} QA pairs. Format: Each line is {{"question": "...", "answer": "..."}}"""
                }
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        
        text = response.choices[0].message.content
        qa_pairs = []
        for line in text.split('\n'):
            line = line.strip()
            if line and line.startswith('{'):
                try:
                    pair = json.loads(line)
                    if "question" in pair and "answer" in pair:
                        qa_pairs.append(pair)
                except json.JSONDecodeError:
                    pass
        return qa_pairs
    except Exception as e:
        print(f"  Error with OpenAI: {e}")
        return []


def generate_qa_pairs(chunks: list[dict], num_questions: int = 3, llm_backend: str = "auto") -> list[dict]:
    """
    Generate QA pairs for all chunks.
    llm_backend: "ollama", "huggingface", "openai", or "auto" to try them in order.
    """
    qa_dataset = []
    backends = {
        "ollama": generate_qa_ollama,
        "huggingface": generate_qa_huggingface,
        "openai": generate_qa_openai,
    }

    for idx, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", idx)
        chunk_text = chunk.get("text", "")

        if not chunk_text.strip():
            print(f"Skipping empty chunk {chunk_id}")
            continue

        print(f"\nProcessing chunk {chunk_id} ({idx+1}/{len(chunks)})...")
        print(f"  Chunk text: {chunk_text[:100]}...")

        qa_pairs = []

        # Try backends in order if auto-detect
        if llm_backend == "auto":
            for backend_name in ["ollama", "huggingface", "openai"]:
                backend_func = backends[backend_name]
                print(f"  Trying {backend_name}...")
                qa_pairs = backend_func(chunk_text, num_questions)
                if qa_pairs:
                    print(f"  Generated {len(qa_pairs)} QA pairs with {backend_name}")
                    llm_backend = backend_name  # Use successful backend for next chunk
                    break
        else:
            if llm_backend in backends:
                backend_func = backends[llm_backend]
                qa_pairs = backend_func(chunk_text, num_questions)
                if qa_pairs:
                    print(f"  Generated {len(qa_pairs)} QA pairs")

        if not qa_pairs:
            print(f"  WARNING: No QA pairs generated for chunk {chunk_id}")

        for qa in qa_pairs:
            qa_dataset.append({
                "chunk_id": chunk_id,
                "section": chunk.get("section", ""),
                "item": chunk.get("item", ""),
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
            })

    return qa_dataset


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate QA pairs from chunks")
    parser.add_argument("--input", default="chunks_10k.json", help="Input chunks file")
    parser.add_argument("--output", default="qa_10k.json", help="Output QA file")
    parser.add_argument("--num-questions", type=int, default=3, help="Questions per chunk")
    parser.add_argument("--backend", default="auto", choices=["auto", "ollama", "huggingface", "openai"],
                        help="LLM backend to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit chunks to process (for testing)")
    
    args = parser.parse_args()

    # Load chunks
    print(f"Loading chunks from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    if args.limit:
        chunks = chunks[:args.limit]
        print(f"Processing first {args.limit} chunks")
    
    print(f"Loaded {len(chunks)} chunks")

    # Generate QA pairs
    print(f"\nGenerating QA pairs using {args.backend}...")
    qa_dataset = generate_qa_pairs(chunks, num_questions=args.num_questions, llm_backend=args.backend)

    # Save output
    print(f"\nSaving {len(qa_dataset)} QA pairs to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, indent=2, ensure_ascii=False)

    print(f"✓ Done! Generated {len(qa_dataset)} QA pairs")
    print(f"\nSample QA pair:")
    if qa_dataset:
        sample = qa_dataset[0]
        print(f"  Chunk ID: {sample['chunk_id']}")
        print(f"  Question: {sample['question']}")
        print(f"  Answer: {sample['answer'][:200]}...")


if __name__ == "__main__":
    main()
