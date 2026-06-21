import re
import json
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

# ── 1. TOKEN UTILITIES ────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def get_tail_tokens(text: str, overlap_tokens: int) -> str:
    """Returns the last N tokens of text as a string (for overlap carry-over)."""
    tokens = enc.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    return enc.decode(tokens[-overlap_tokens:])


# ── 2. CLEAN PARAGRAPHS ───────────────────────────────────────────────────────

def clean_paragraphs(paragraphs: list[str]) -> list[str]:
    filtered = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        # Drop only truly empty or single-word lines
        if len(p.split()) < 3:
            continue

        # Drop lines that are ONLY punctuation/whitespace — no alphanumeric at all
        if not re.search(r"[a-zA-Z0-9]", p):
            continue

        # Drop pure page numbers — just a standalone number, nothing else
        if re.fullmatch(r"\d{1,3}", p.strip()):
            continue
        
        # Drop table-like content: rows with many pipes/separators or all-numeric rows
        # e.g., "1,234 | 5,678 | 9,101" or "2025 | 2024 | 2023"
        if re.match(r"^[\s\d,\.\-\$\%\|\–—]+$", p):
            continue

        filtered.append(p)

    # Merge short lines (3–15 words) into the next substantive paragraph
    # These are sub-labels like "Results of Operations" or "Operating margin: 45.6%"
    merged = []
    buffer = ""
    for p in filtered:
        word_count = len(p.split())
        if word_count <= 15:
            buffer = (buffer + " " + p).strip() if buffer else p
        else:
            full = (buffer + " " + p).strip() if buffer else p
            buffer = ""
            merged.append(full)

    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    return merged


# ── 3. SPLIT LONG PARAGRAPH AT SENTENCE BOUNDARIES ───────────────────────────

def _split_text_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Split a text block into chunks that each fit within max_tokens."""
    words = text.split()
    if not words:
        return []

    chunks = []
    current = []
    for word in words:
        word_tokens = count_tokens(word)
        if word_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
            tokens = enc.encode(word)
            for i in range(0, len(tokens), max_tokens):
                chunks.append(enc.decode(tokens[i:i + max_tokens]))
            continue

        candidate = " ".join(current + [word]) if current else word
        if count_tokens(candidate) > max_tokens:
            if current:
                chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(" ".join(current))
    return chunks


def _find_natural_split_index(text: str) -> int:
    # Prefer split points at sentence-like boundaries or table row starts.
    patterns = [
        r'(?<=[.!?;:])\s+(?=[A-Z0-9"\'\(\[\“\‘])',
        r'(?<=\))\s+(?=[A-Z])',
        r'(?<=\$)\s+(?=[A-Z])',
        r'(?<=\d)\s+(?=[A-Z][a-z])',
        r'(?<=\S)\s+(?=[A-Z])',
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            return matches[-1].end()
    return text.rfind(' ')


def _split_long_sentence_by_tokens(sentence: str, max_tokens: int) -> list[str]:
    tokens = enc.encode(sentence)
    if len(tokens) <= max_tokens:
        return [sentence.strip()]

    parts = []
    index = 0
    while index < len(tokens):
        end = min(index + max_tokens, len(tokens))
        chunk_text = enc.decode(tokens[index:end]).strip()

        if end < len(tokens):
            split_idx = _find_natural_split_index(chunk_text)
            if split_idx > 0:
                part = chunk_text[:split_idx].strip()
                part_tokens = len(enc.encode(part))
                if part_tokens < 1:
                    part = chunk_text
                    part_tokens = len(enc.encode(part))
            else:
                part = chunk_text
                part_tokens = len(enc.encode(part))
        else:
            part = chunk_text
            part_tokens = len(enc.encode(part))

        parts.append(part)
        if part_tokens == 0:
            break
        index += part_tokens

    return parts


def split_into_sentences(text: str) -> list[str]:
    # Only split at sentence-ending punctuation when the next token
    # looks like the start of a new sentence. This avoids splitting on
    # abbreviations such as "U.S." followed by a lowercase continuation.
    boundary = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[\“\‘])')
    sentences = boundary.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def get_tail_sentences(sentences: list[str], overlap_tokens: int) -> list[str]:
    tail = []
    total = 0
    for sent in reversed(sentences):
        sent_tokens = count_tokens(sent)
        if total + sent_tokens > overlap_tokens and tail:
            break
        tail.insert(0, sent)
        total += sent_tokens
    return tail


def split_by_sentence(text: str,
                      max_tokens: int,
                      carry_over: str = "") -> list[tuple[str, str]]:
    """
    Splits a single oversized paragraph at sentence boundaries.
    carry_over is prepended to the first chunk for context continuity.
    Returns tuples of (chunk_text, source).
    """
    sentences = split_into_sentences(text)
    carry_sentences = split_into_sentences(carry_over) if carry_over else []
    chunks = []

    current = carry_sentences.copy()
    current_tokens = count_tokens(" ".join(current)) if current else 0

    def append_current_chunk(source: str = "sentence_split"):
        nonlocal current, current_tokens
        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append((chunk_text, source))
            current = carry_sentences.copy() if carry_sentences else []
            current_tokens = count_tokens(" ".join(current)) if current else 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)
        if current_tokens and current_tokens + sent_tokens > max_tokens:
            append_current_chunk()

        if sent_tokens > max_tokens:
            parts = _split_long_sentence_by_tokens(sent, max_tokens)
            for part in parts:
                part_tokens = count_tokens(part)
                if current_tokens and current_tokens + part_tokens > max_tokens:
                    append_current_chunk("long_sentence_split")
                current.append(part)
                current_tokens = count_tokens(" ".join(current))
                if current_tokens >= max_tokens:
                    append_current_chunk("long_sentence_split")
            continue

        if not current and sent_tokens <= max_tokens:
            current.append(sent)
            current_tokens = sent_tokens
        elif current:
            current.append(sent)
            current_tokens = count_tokens(" ".join(current))

    if current:
        chunk_text = " ".join(current).strip()
        if chunk_text:
            chunks.append((chunk_text, "sentence_split"))

    final_chunks = []
    for chunk_text, source in chunks:
        if count_tokens(chunk_text) <= max_tokens:
            final_chunks.append((chunk_text, source))
        else:
            for split_text in _split_text_by_tokens(chunk_text, max_tokens):
                final_chunks.append((split_text, source))

    return final_chunks


# ── 4. CORE: CHUNK ONE SECTION ────────────────────────────────────────────────

def chunk_section(section: dict,
                  max_tokens: int = 300,
                  overlap_tokens: int = 75) -> list[dict]:
    """
    Chunks a single parsed section using sliding window over sentences.

    Strategy:
    - Split paragraphs into sentences and chunk whole sentences only
    - Accumulate sentences until max_tokens is reached
    - Overlap only full sentences between chunks
    - If a single sentence exceeds max_tokens, split it by tokens
    - Every chunk tagged with section, token count, and source type
    """
    raw_paragraphs = section.get("paragraphs", [])
    section_label  = section.get("full", "Unknown Section")
    item_label     = section.get("item", "")
    title          = section.get("title", "")

    paragraphs = clean_paragraphs(raw_paragraphs)
    if not paragraphs:
        return []

    chunks = []
    buffer = []
    buf_tokens = 0
    carry_over_sentences = []

    def flush_buffer(source: str):
        nonlocal carry_over_sentences, buffer, buf_tokens
        if not buffer:
            return
        current_sentences = carry_over_sentences + buffer
        text = " ".join(current_sentences).strip()
        chunks.append(_make_chunk(text, section_label, item_label, title, source))
        carry_over_sentences = get_tail_sentences(current_sentences, overlap_tokens)
        buffer = []
        buf_tokens = 0

    for para in paragraphs:
        sentences = split_into_sentences(para)
        for sent in sentences:
            sent_tokens = count_tokens(sent)
            if sent_tokens > max_tokens:
                flush_buffer("sentence_window")
                parts = _split_long_sentence_by_tokens(sent, max_tokens)
                for part in parts:
                    chunks.append(_make_chunk(part, section_label, item_label, title, "sentence_split"))
                carry_over_sentences = get_tail_sentences(parts, overlap_tokens)
                continue

            carry_tokens = count_tokens(" ".join(carry_over_sentences)) if carry_over_sentences else 0
            if buf_tokens + sent_tokens + carry_tokens > max_tokens and buffer:
                flush_buffer("sentence_window")

            buffer.append(sent)
            buf_tokens += sent_tokens

    flush_buffer("sentence_window")
    return chunks


def _make_chunk(text: str,
                section: str,
                item: str,
                title: str,
                source: str) -> dict:
    text = text.strip()
    return {
        "section":  section,
        "item":     item,
        "title":    title,
        "text":     text,
        "tokens":   count_tokens(text),
        "source":   source,   # "paragraph_window" | "sentence_split"
    }


# ── 5. CHUNK ALL SECTIONS ─────────────────────────────────────────────────────

def chunk_all_sections(sections: list[dict],
                       max_tokens: int = 300,
                       overlap_tokens: int = 75) -> list[dict]:
    """
    Runs chunking across all parsed sections.
    Adds a global chunk_id for easy reference during QA generation.
    """
    all_chunks = []
    chunk_id   = 0

    for section in sections:
        section_chunks = chunk_section(section, max_tokens, overlap_tokens)
        for chunk in section_chunks:
            chunk["chunk_id"] = chunk_id
            all_chunks.append(chunk)
            chunk_id += 1

    return all_chunks


# ── 6. DIAGNOSTICS ────────────────────────────────────────────────────────────

def diagnose_chunks(chunks: list[dict]):
    """Prints a distribution summary so you know if tuning is needed."""
    if not chunks:
        print("No chunks to diagnose.")
        return

    tokens = [c["tokens"] for c in chunks]
    by_source = {}
    for c in chunks:
        by_source.setdefault(c["source"], []).append(c["tokens"])

    print(f"\n── CHUNK DIAGNOSTICS ──────────────────────────────────")
    print(f"  Total chunks     : {len(chunks)}")
    print(f"  Token range      : {min(tokens)} – {max(tokens)}")
    print(f"  Average tokens   : {sum(tokens) // len(tokens)}")
    print(f"  Median tokens    : {sorted(tokens)[len(tokens)//2]}")
    print(f"  Too short (<30)  : {sum(1 for t in tokens if t < 30)}")
    print(f"  Ideal (30–350)   : {sum(1 for t in tokens if 30 <= t <= 350)}")
    print(f"  Too long (>350)  : {sum(1 for t in tokens if t > 350)}")

    print(f"\n  By source:")
    for src, src_tokens in by_source.items():
        print(f"    {src:25s} → {len(src_tokens)} chunks, "
              f"avg {sum(src_tokens)//len(src_tokens)} tokens")

    print(f"\n  By section:")
    by_section = {}
    for c in chunks:
        by_section.setdefault(c["section"], 0)
        by_section[c["section"]] += 1
    for sec, count in by_section.items():
        print(f"    {sec[:55]:55s} → {count} chunks")

    # Show a few examples of suspiciously short chunks for inspection
    short = [c for c in chunks if c["tokens"] < 30]
    if short:
        print(f"\n  ⚠ Short chunk samples:")
        for c in short[:3]:
            print(f"    [{c['section']}] \"{c['text'][:80]}\"")


# ── 7. RUN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load your parser output
    with open("parsed_10k_2.json", "r", encoding="utf-8") as f:
        sections = json.load(f)

    # Chunk
    chunks = chunk_all_sections(
        sections,
        max_tokens    = 300,   # target chunk size
        overlap_tokens = 75    # ~3-4 sentences of cross-paragraph context
    )

    # Diagnose before saving
    diagnose_chunks(chunks)

    # Save
    with open("chunks_10k.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(chunks)} chunks to chunks_10k.json")

    # Preview first 3
    print("\n── PREVIEW ──────────────────────────────────────────────")
    for c in chunks[:3]:
        print(f"\nchunk_id : {c['chunk_id']}")
        print(f"section  : {c['section']}")
        print(f"source   : {c['source']}")
        print(f"tokens   : {c['tokens']}")
        print(f"text     : {c['text'][:300]}...")