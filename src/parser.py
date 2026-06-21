import re
import json
from bs4 import BeautifulSoup

# ── 1. CLEAN HTML ─────────────────────────────────────────────────────────────
def clean_html(filepath: str) -> BeautifulSoup:
    with open(filepath, "rb") as f:
        raw = f.read()
    try:
        import chardet
        enc = chardet.detect(raw)["encoding"] or "utf-8"
    except ImportError:
        enc = "utf-8"

    soup = BeautifulSoup(raw.decode(enc, errors="replace"), "lxml")
    for tag in soup(["script", "style", "noscript", "meta",
                     "link", "ix:header", "ix:hidden"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()
    return soup


# ── 2. EXTRACT TOC DYNAMICALLY ───────────────────────────────────────────────
def parse_toc_label_title(text: str):
    """Parse an Item label and title from a candidate TOC text line."""
    text = re.sub(r"\s+", " ", text).strip()
    match = re.match(r"^(item\s*\d+[A-Za-z]?)[\.\-–—]?\s*(.+)$", text, re.I)
    if not match:
        return None

    label = match.group(1).strip()
    remainder = match.group(2).strip()
    # Remove trailing page numbers or page labels
    remainder = re.sub(r"(?:page\s*)?\d+\s*$", "", remainder, flags=re.I).strip()
    if not remainder or remainder.lower() == label.lower():
        return None
    if re.fullmatch(r"[\W_]+", remainder):
        return None

    return label, remainder


def extract_toc(soup: BeautifulSoup) -> list[dict]:
    """
    Finds the Table of Contents by looking for clusters of links
    and plain text rows that match the SEC Item pattern.
    Returns: [{"label": "Item 1A", "title": "Risk Factors", "anchor": "item1a"}, ...]
    """
    # Pattern: "Item 1", "Item 1A", "Item 1B", "Item 2", etc.
    ITEM_RE = re.compile(r"^item\s*\d+[a-zA-Z]?\b", re.I)

    toc_entries = {}

    # Strategy 1: find all <a> tags whose text starts with "Item X"
    for a in soup.find_all("a", href=True):
        text = a.get_text(separator=" ", strip=True)
        if not ITEM_RE.match(text):
            continue

        parts = re.split(r"[.\-–—\n]\s*", text, maxsplit=1)
        label = parts[0].strip()
        title = parts[1].strip() if len(parts) > 1 else ""

        key = label.lower().replace(" ", "")
        if key in toc_entries:
            continue

        href = a["href"]
        anchor = href.lstrip("#") if href.startswith("#") else None

        toc_entries[key] = {
            "label": label,
            "title": title,
            "anchor": anchor,
            "full": f"{label} – {title}" if title else label
        }

    # Strategy 2: fallback to plain text rows / cells to fill missing titles or missing entries
    for tag in soup.find_all(["tr", "p", "li", "td", "th"]):
        text = tag.get_text(separator=" ", strip=True)
        if not ITEM_RE.search(text):
            continue

        parsed = parse_toc_label_title(text)
        if not parsed:
            continue
        label, title = parsed
        if not title:
            continue

        key = label.lower().replace(" ", "")
        if key in toc_entries:
            existing = toc_entries[key]
            if not existing["title"]:
                existing["title"] = title
                existing["full"] = f"{label} – {title}"
            continue

        toc_entries[key] = {
            "label": label,
            "title": title,
            "anchor": None,
            "full": f"{label} – {title}"
        }

    return list(toc_entries.values())


def dedup_blocks(texts: list[str], similarity_threshold: float = 0.85) -> list[str]:
    """
    Removes near-duplicate paragraphs that EDGAR tables leave behind
    as ghost text in surrounding divs.
    """
    def token_set(t: str) -> set:
        return set(re.findall(r"\w+", t.lower()))

    def similarity(a: str, b: str) -> float:
        sa, sb = token_set(a), token_set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    deduped = []
    for text in texts:
        # Compare against last 3 kept blocks (not just the immediately previous one)
        too_similar = any(
            similarity(text, prev) >= similarity_threshold
            for prev in deduped[-3:]
        )
        if not too_similar:
            deduped.append(text)

    return deduped

def extract_section_content(start_tag, end_tag, section_name: str, next_label: str = None) -> dict:
    text_blocks = []

    for tag in start_tag.find_all_next():
        if start_tag in tag.parents:
            continue

        if end_tag and (tag is end_tag or end_tag in tag.descendants):
            break

        text = tag.get_text(separator=" ", strip=True)
        if next_label and re.match(rf"^{re.escape(next_label)}(?:[\.\s\-–—]|$)", text, re.I):
            break

        # ── Skip all table content ──────────────────────────────────────────
        if tag.find_parent("table"):
            continue
        if tag.name in ["table", "tr", "td", "th"]:
            continue

        # ── Only keep real paragraph-level elements ─────────────────────────
        if tag.name not in ["p", "div", "li", "h1", "h2", "h3", "h4", "h5"]:
            continue

        # ── Quality filters ─────────────────────────────────────────────────
        if len(text) < 15:
            continue
        if not re.search(r"[a-zA-Z]", text):   # must have at least one letter
            continue

        text_blocks.append(text)

    return {
        "section":     section_name,
        "text_blocks": dedup_blocks(text_blocks),
    }


# ── 3. FIND ANCHOR ELEMENTS IN DOCUMENT ──────────────────────────────────────
def get_block_parent(tag):
    if tag is None:
        return None
    block_tags = {"p", "div", "td", "th", "tr", "table", "li", "section", "article", "body"}
    while tag is not None and tag.name not in block_tags:
        tag = tag.parent
    return tag


def find_anchor_tag(soup: BeautifulSoup, anchor: str = None, label: str = None, title: str = None):
    """
    Finds the actual element in the document where a section begins.
    Tries multiple strategies since EDGAR filings vary.
    """
    if anchor:
        tag = soup.find("a", attrs={"name": re.compile(f"^{re.escape(anchor)}$", re.I)})
        if tag:
            return get_block_parent(tag)

        tag = soup.find(id=re.compile(f"^{re.escape(anchor)}$", re.I))
        if tag:
            return get_block_parent(tag)

        clean = re.sub(r"[-_]", "", anchor).lower()
        tag = soup.find("a", attrs={"name": re.compile(clean, re.I)})
        if tag:
            return get_block_parent(tag)

    def normalize_text(value: str) -> str:
        return re.sub(r"[^0-9a-z]+", "", (value or "").lower())

    if label and title:
        target = normalize_text(f"{label} {title}")
        candidates = []
        for tag in soup.find_all(["p", "div", "td", "th", "li", "span", "section", "article"]):
            text = normalize_text(tag.get_text(" ", strip=True))
            if target in text:
                candidates.append(tag)
        if candidates:
            return candidates[-1]

    if label:
        search = re.compile(rf"^{re.escape(label)}(?:[\.\s\-–—]|$)", re.I)
        candidates = []
        for string in soup.find_all(string=search):
            parent = get_block_parent(string.parent)
            if parent is None:
                continue
            candidates.append(parent)
        if candidates:
            return candidates[-1]

    return None


# ── 4. EXTRACT TEXT AFTER AN ANCHOR UNTIL NEXT ANCHOR ────────────────────────
def extract_between(start_tag, end_tag, next_label: str = None) -> list[str]:
    """
    Collects all text-bearing elements between start_tag and end_tag
    in document order.
    """
    collecting = False
    texts = []

    # Walk all tags in document order
    for tag in start_tag.find_all_next():
        if start_tag in tag.parents:
            continue

        # Stop when we hit the next section anchor or the block containing it
        if end_tag and (tag is end_tag or end_tag in tag.descendants):
            break

        text = tag.get_text(separator=" ", strip=True)
        if next_label and re.match(rf"^{re.escape(next_label)}(?:[\.\s\-–—]|$)", text, re.I):
            break

        # Skip ALL table-related content — check parent before name
        if tag.find_parent("table"):
            continue

        if tag.name in ["table", "tr", "td", "th"]:
            continue

        if tag.name not in ["p", "div", "li", "h1", "h2", "h3", "h4", "h5"]:
            continue

        text = tag.get_text(separator=" ", strip=True)

        # Quality filters
        if len(text) < 15:
            continue
        # Only drop if zero alphabetic characters — keeps "$281.7 billion" etc.
        if not re.search(r"[a-zA-Z]", text):
            continue

        texts.append(text)

    # Deduplicate consecutive near-duplicates (EDGAR has lots of these)
    deduped = dedup_blocks(texts)

    return deduped


# ── 5. CHUNK A SECTION ────────────────────────────────────────────────────────
def chunk_texts(texts: list[str], max_tokens: int = 500) -> list[str]:
    chunks, current, current_len = [], [], 0
    for text in texts:
        wc = len(text.split())
        if current_len + wc > max_tokens and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(text)
        current_len += wc
    if current:
        chunks.append(" ".join(current))
    return chunks


# ── 6. MAIN PIPELINE ──────────────────────────────────────────────────────────
def parse_10k_dynamic(filepath: str, max_tokens: int = 500) -> dict:
    print("[1/4] Cleaning HTML...")
    soup = clean_html(filepath)

    print("[2/4] Extracting Table of Contents...")
    toc = extract_toc(soup)
    if not toc:
        print("      ⚠ No TOC found — check if file has anchor links")
        return {}
    print(f"      → {len(toc)} sections found in TOC:")
    for entry in toc:
        print(f"         {entry['full']}  (anchor: {entry['anchor']})")

    print("[3/4] Resolving anchors and extracting sections...")
    # Resolve each TOC entry to its anchor tag in the document
    resolved = []
    for entry in toc:
        tag = find_anchor_tag(soup, entry.get("anchor"), entry["label"], entry.get("title"))
        resolved.append({**entry, "tag": tag})
        status = "✓" if tag else "✗ not found"
        print(f"         {entry['label']}: {status}")

    print("[4/4] Collecting section content...")
    sections = []
    for i, entry in enumerate(resolved):
        if not entry["tag"]:
            continue
        next_tag = next(
            (r["tag"] for r in resolved[i+1:] if r["tag"]), None
        )
        next_label = next((r["label"] for r in resolved[i+1:] if r["tag"]), None)
        section_data = extract_section_content(entry["tag"], next_tag, entry["full"], next_label)
        text = "\n\n".join(section_data["text_blocks"]).strip()
        sections.append({
            "item": entry["label"],
            "title": entry["title"],
            "full": entry["full"],
            "paragraphs": section_data["text_blocks"],
            "text": text,
        })
        print(f"      → {entry['full']}: {len(section_data['text_blocks'])} paragraphs")

    return sections


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    output = parse_10k_dynamic(r"C:\Users\HP\Documents\Caliperlab\data\msft-20250630.htm")

    with open("parsed_10k_2.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n── PREVIEW ──")
    for section in output[:3]:
        print(f"\n{'='*60}")
        print(f"SECTION: {section['full']}")
        print(f"TEXT: {section['text'][:400]}...")
