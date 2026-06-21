# Financial QA Dataset Generation from SEC 10-K Filings

## Overview

This project generates structured question-answer datasets from SEC 10-K filings. The pipeline extracts meaningful content from filing HTML, identifies major SEC sections, creates context-preserving chunks, generates QA pairs using a Large Language Model (LLM), and applies validation and deduplication before producing the final dataset.

The design prioritizes:

- Grounding to source text
- Traceability to filing sections
- Cost-efficient generation
- Modular and extensible processing

### Main Components

- `src/parser.py`
  - HTML cleaning
  - Dynamic Table of Contents extraction
  - Section resolution
  - Paragraph extraction and deduplication

- `src/chunker.py`
  - Paragraph cleaning
  - Sentence-aware chunking
  - Token-aware splitting
  - Context overlap management

- `src/qa_generator.py`
  - QA generation using Groq
  - Grounding verification
  - Quality filtering
  - Dataset deduplication

---

## Pipeline

### 1. HTML Cleaning

The pipeline begins by parsing raw SEC filing HTML using BeautifulSoup with the `lxml` parser.

Before extraction, non-content elements are removed, including:

- Scripts
- Stylesheets
- Hidden content
- Navigation metadata
- Inline XBRL tags (`ix:header`, `ix:hidden`)

Character encoding is automatically detected using `chardet` to improve robustness across filings with different encodings.

---

### 2. Dynamic Table of Contents Extraction

Major SEC filing sections are identified dynamically instead of relying on hardcoded section positions.

The parser detects section labels such as:

- Item 1
- Item 1A
- Item 1B
- Item 7
- Item 8

Two complementary extraction strategies are used:

1. Anchor-based extraction from hyperlinks.
2. Plain-text extraction from rows, paragraphs, and list elements.

This improves compatibility across different filing formats.

---

### 3. Section Resolution

Each discovered TOC entry is mapped to its corresponding location in the filing.

To improve robustness, the parser attempts multiple resolution strategies:

- Exact anchor matching
- HTML ID matching
- Normalized anchor matching
- Label-title matching
- Label-only fallback matching

This approach increases the likelihood of correctly identifying section boundaries even when filings use inconsistent markup.

---

### 4. Paragraph Extraction and Cleaning

After section boundaries are identified, content is extracted between consecutive sections.

The extraction process:

- Removes all table content
- Filters short or low-quality text
- Removes formatting artifacts
- Keeps only paragraph-level content

Supported content blocks include:

- Paragraphs (`p`)
- Divisions (`div`)
- List items (`li`)
- Headings (`h1`–`h5`)

---

### 5. Paragraph Deduplication

SEC filings frequently contain duplicated content due to EDGAR formatting and nested HTML structures.

To reduce redundancy, extracted text blocks are compared using token-overlap similarity.

Near-duplicate paragraphs are removed before chunk generation, producing cleaner source material for downstream QA generation.

---

### 6. Context-Preserving Chunking

Long filing sections are divided into smaller chunks suitable for LLM processing.

The chunking stage uses:

- Token-aware sizing via `tiktoken`
- Sentence boundary preservation
- Sliding window chunk construction
- Configurable context overlap

Default configuration:

- Maximum chunk size: 300 tokens
- Overlap: 75 tokens

This balances context availability with generation quality.

---

### 7. Long-Sentence Handling

Some financial disclosures contain unusually long sentences that exceed chunk limits.

When necessary, the chunker performs controlled token-based splitting while attempting to preserve natural language boundaries.

This prevents oversized prompts without discarding information.

---

### 8. QA Generation

Each chunk is independently processed to generate grounded question-answer pairs.

### Model Selection

The generation stage uses:

`meta-llama/llama-4-scout-17b-16e-instruct`

accessed through the Groq API.

This model was selected because it provided the strongest balance of:

- Generation quality
- Context understanding
- Reliability
- Token availability

among the models available during development.

The model consistently produced grounded QA pairs while remaining practical for large-scale dataset generation.

---

### 9. Structured Output Generation

For each chunk, the model generates up to two question-answer pairs.

Each pair includes:

- Question
- Answer
- Supporting source sentence
- Question type
- Difficulty label

Supported question types:

- Fact Extraction
- Numeric Calculation
- Comparison
- Multi-Step Reasoning

The model is instructed to return JSON-lines output to simplify parsing and downstream processing.

---

### 10. Rule-Based Verification

A separate LLM verification stage was intentionally avoided to reduce inference cost and improve throughput.

Instead, generated QA pairs are validated using deterministic checks.

Verification consists of:

#### Source Grounding Validation

Every QA pair must provide a source sentence.

The source is accepted only if:

- It appears directly in the original chunk, or
- It achieves a high token-overlap similarity score

#### Clone Detection

Rejects cases where:

- The answer duplicates the source verbatim
- The question is embedded inside the answer

#### Question Quality Validation

Questions must:

- Contain meaningful interrogative structure
- Meet minimum length requirements
- Use valid question forms

Examples include:

- What
- Why
- How
- Which
- When
- Who

Only QA pairs that pass all verification checks are retained.

---

### 11. Dataset Deduplication

Generated QA pairs are deduplicated using normalized question-answer matching.

This removes repeated outputs generated from overlapping chunks and semantically similar sections.

The result is a cleaner and more diverse dataset.

---

## Usage

### Step 1: Parse Filing

```powershell
python src/parser.py
```

Output:

```text
parsed_10k_2.json
```

---

### Step 2: Generate Chunks

```powershell
python src/chunker.py
```

Output:

```text
chunks_10k.json
```

---

### Step 3: Generate QA Dataset

```powershell
python src/qa_generator.py --input chunks_10k.json --output qa_10k.json --rejected qa_rejected.json --csv
```

---

## Design Choices

### Why BeautifulSoup with lxml?

SEC filings often contain malformed and inconsistent HTML.

BeautifulSoup provides a robust DOM-based parsing approach while the `lxml` backend offers efficient processing of large filings.

This combination improves reliability compared to regex-based extraction.

### Why Dynamic TOC Extraction?

Different filings use different formatting conventions.

Dynamic TOC discovery avoids dependence on filing-specific templates and improves generalization across companies and filing years.

### Why Remove Tables?

Financial tables often contain highly structured layouts that are difficult to convert into clean natural-language context.

The current pipeline focuses on narrative disclosures and management discussion sections, where question-answer generation is most effective.

### Why Sentence-Based Chunking?

Splitting text arbitrarily can break important context.

Sentence-aware chunking preserves semantic structure and improves grounding quality during generation.

### Why Overlapping Chunks?

Important information frequently appears near chunk boundaries.

A sliding-window overlap preserves contextual continuity and reduces information loss between neighboring chunks.

### Why Use Llama 4 Scout?

The selected model provided the best trade-off between:

- Output quality
- Reliability
- Context handling
- Token availability

while remaining practical for large-scale generation workloads.

### Why Rule-Based Verification?

Using a second LLM verifier significantly increases latency and token consumption.

Rule-based validation provides:

- Faster execution
- Lower cost
- Deterministic behavior

while still filtering many unsupported or low-quality outputs.

### Why Preserve Metadata?

Every QA pair retains filing, section, item, and chunk metadata.

This improves traceability and enables future retrieval, filtering, evaluation, and dataset analysis workflows.

### Why a Modular Architecture?

The pipeline is separated into parsing, chunking, generation, verification, and deduplication stages.

This simplifies debugging and allows individual components to be improved independently without affecting the rest of the system.

---

## Output Format

Each verified QA pair contains:

```json
{
  "chunk_id": 0,
  "filing": "chunks_10k",
  "section": "Item 1A – Risk Factors",
  "item": "Item 1A",
  "question": "...",
  "answer": "...",
  "source": "...",
  "type": "fact extraction",
  "difficulty": "medium"
}
```

When CSV export is enabled, the dataset is also saved in CSV format for easier inspection.

---

## Known Limitations

- SEC filings vary significantly in HTML structure, and some section boundaries may not be detected correctly.
- Financial tables are currently excluded from extraction, causing some numerical disclosures to be omitted.
- Rule-based verification cannot detect all factual inaccuracies.
- Difficulty labels depend on model output rather than formal evaluation.
- Semantic duplicates using different wording may occasionally survive deduplication.
- The current implementation processes one filing at a time.
- Human review is still recommended before using generated datasets for benchmarking or production applications.

---

## Scaling to Multiple Documents or 1000+ QA Pairs

To scale the pipeline, the following improvements can be applied:

1. Process filings independently and merge outputs using filing metadata.
2. Batch generation requests to improve throughput.
3. Cache generated outputs using chunk hashes.
4. Parallelize chunk processing when API limits allow.
5. Persist intermediate artifacts for resumability.
6. Monitor acceptance and rejection statistics.
7. Introduce embedding-based semantic deduplication.
8. Deploy distributed worker pipelines for large filing collections.

These changes would enable efficient generation of thousands of grounded QA pairs while maintaining quality and traceability.

---

## Notes

- This repository is designed as a research-oriented data generation pipeline rather than a production ingestion platform.
- The primary goal is generating grounded, traceable QA datasets from long-form financial disclosures.
- The design emphasizes transparency, reproducibility, and modularity over maximum throughput.