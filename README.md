# Financial QA Dataset Generation from SEC 10-K Filings

## Overview

This pipeline converts SEC 10-K filings into structured question-answer datasets.
It parses filing HTML, extracts major sections, splits text into chunks, and uses an LLM to generate QA pairs.
A verification pass then filters and deduplicates the output before saving the final dataset.

The main implementation files are:
- `src/parser.py` → HTML cleanup, TOC extraction, section resolution, paragraph extraction, chunking
- `src/chunker.py` → chunk generation from parsed paragraphs
- `src/qa_generator.py` → Groq-backed QA generation, grounding verification, deduplication


## Pipeline

1. Clean raw 10-K HTML
   - Remove scripts, styles, hidden XBRL metadata, and other non-content tags
   - Keep visible paragraphs and headings

2. Extract the Table of Contents
   - Detect SEC item labels such as `Item 1`, `Item 1A`, `Item 7`, `Item 8`
   - Use anchors and surrounding text to locate section starts

3. Resolve section anchors
   - Map TOC entries to actual content in the filing
   - Fall back from anchor links to text matching when needed

4. Extract section paragraphs
   - Collect paragraph-like blocks (`p`, `div`, `li`, headings)
   - Filter out tables and low-quality text
   - Deduplicate near-duplicate blocks introduced by EDGAR page structure

5. Chunk text for generation
   - Combine paragraphs into chunks of roughly 400–500 words
   - Preserve paragraph boundaries within chunks
   - Avoid overly large prompts that reduce generation quality

6. Generate QA pairs
   - Use an LLM backend to generate QA pairs per chunk
   - Backend examples:
     - Groq (`src/qa_generator.py`)
   - Prompt the model to return JSON-lines output for structured parsing

7. Verify and deduplicate
   - Ground each QA pair against the source chunk
   - Reject pairs with missing source grounding, low-quality questions, or cloning artifacts
   - Remove duplicate question-answer pairs

## Usage

Run the parser to extract sections and paragraphs first:

```powershell
python src/parser.py
```

This will produce a parsed JSON file such as `parsed_10k_2.json`.

Run the chunker to get smaller portions of text:

```powershell
python src/chunker.py
```

This will produce a chunked JSON file such as `chunks_10k.json`.

Generate QA pairs from chunked data:

```powershell
python src/qa_generator.py --input chunks_10k.json --output qa_10k.json --rejected qa_rejected.json --csv
```

Or specify a different LLM backend:

```powershell
python src/qa_generator.py --input chunks_10k.json --output qa_10k.json --backend openai
```

## Design Choices

### Why parse by section and chunk text?

- SEC filings are long and semi-structured. Section-aware parsing keeps QA generation aligned with the document's natural hierarchy.
- Chunking into paragraph-aware windows reduces prompt length and helps the LLM stay grounded in a small context.

### Why separate generation from verification?

- Generation can produce plausible but unsupported answers.
- A second pass validates the source quote and rejects obvious failures without repeating generation costs.
- This keeps the final dataset cleaner and easier to inspect.

### Why support multiple backends?

- Local or remote LLM backends provide flexibility when Groq is unavailable.
- Hugging Face and OpenAI support remote inference and provide fallback options when a local model is unavailable.

## Output Format

The final dataset includes fields such as:
- `chunk_id`
- `filing`
- `section`
- `item`
- `question`
- `answer`
- `source`
- `type`
- `difficulty`

When CSV export is enabled, the repository also writes a `.csv` file for easier review.

## Known Limitations

- HTML parsing is brittle. SEC filings vary widely in markup, and section detection may fail for non-standard filings.
- Table content is ignored by the current extractor, so numeric tables and structured disclosures are often lost.
- LLM outputs still may hallucinate or generate partial answers even after verification.
- Difficulty labels and question types are heuristic and depend on the model's output.
- Rule-based verification is not a substitute for human review; it catches simple grounding issues but not subtle factual errors.
- The current pipeline assumes one input filing at a time and does not yet include distributed batch orchestration.

## Scaling to Multiple Documents or 1000+ QA Pairs

To scale this design, I would:

1. Process documents independently
   - Parse each filing into a separate chunk file
   - Preserve filing metadata so outputs can be merged later

2. Use batched generation
   - Keep chunk generation in batches of 20–50 chunks to avoid model overload
   - Write intermediate outputs to disk frequently for resumability

3. Cache generation results
   - Hash each chunk text and save generated QA pairs
   - Skip regeneration for unchanged chunks

4. Parallelize safely
   - Run multiple generator workers if the backend supports concurrent requests
   - Keep verification local and lightweight to avoid repeated LLM calls

5. Optimize prompts and chunk size
   - Use smaller, high-quality chunks for more consistent QA pair grounding
   - Reduce `num_questions` per chunk if the model starts to produce lower-quality outputs

6. Monitor quality and rejection rates
   - Track accepted vs rejected pair counts per document
   - Adjust prompts, chunk size, and backend selection based on observed failures

With these practices, generating 1,000+ high-quality pairs becomes practical by splitting work into smaller, reusable pieces, avoiding monolithic prompts, and storing parse/artifact files for repeatable batch processing.

## Notes

- The repository is intentionally designed as a research-style pipeline, not a production ingestion service.
- The current focus is on grounding quality and traceability rather than on handling every possible SEC filing format.
