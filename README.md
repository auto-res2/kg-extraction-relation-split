# KG Extraction from JacRED: Relation-Split Multi-Pass Extraction

## 1. Background

### JacRED Dataset: Japanese Document-level Relation Extraction Dataset

- **Source**: https://github.com/YoumiMa/JacRED (clone to `/tmp/JacRED`)
- **Splits**: train (1400 docs), dev (300 docs), test (300 docs)
- **Format**: Each document has:
  - `title`: document title
  - `sents`: tokenized sentences (list of list of tokens)
  - `vertexSet`: entities with mentions (list of entity groups, each containing mention dicts with `name`, `type`, `sent_id`, `pos`)
  - `labels`: relations as `{h, t, r, evidence}` where h/t are vertexSet indices and r is a Wikidata P-code
- **9 entity types**: PER, ORG, LOC, ART, DAT, TIM, MON, %, NA
- **35 relation types**: Wikidata P-codes (P131, P27, P569, P570, P19, P20, P40, P3373, P26, P1344, P463, P361, P6, P127, P112, P108, P137, P69, P166, P170, P175, P123, P1441, P400, P36, P1376, P276, P937, P155, P156, P710, P527, P1830, P121, P674)
- **Statistics**: Avg ~17 entities/doc, avg ~20 relations/doc, avg ~253 chars/doc

## 2. Base Implementation (already provided)

The following files implement the baseline and two-stage extraction:

- **run_experiment.py**: Main orchestrator. Loads data, runs conditions (Baseline, Two-Stage), prints comparison table, saves results.json.
- **data_loader.py**: Data loading from JacRED JSON files, document selection (10 stratified from dev split), few-shot example selection, domain/range constraint table construction from training data.
- **llm_client.py**: Gemini API wrapper using `google-genai` library with Structured Outputs (`response_mime_type="application/json"` + `response_schema`), ThinkingConfig, and retry logic.
- **prompts.py**: All prompt templates including system prompt with 35 relation types defined in Japanese, extraction prompt (baseline and recall-oriented modes), and verification prompt for Stage 2.
- **extraction.py**: Two conditions:
  - `run_baseline()`: Single LLM call extraction with post-filtering (invalid labels, invalid entity types).
  - `run_proposed()`: Two-Stage generate+verify. Stage 1 extracts with recall-oriented prompt, Stage 2 batch-verifies candidates, then applies domain/range constraints.
- **evaluation.py**: Entity alignment (3-pass: exact match -> normalized match -> substring match) and micro-averaged P/R/F1 computation.
- **schemas.py**: JSON schemas for Gemini Structured Outputs (extraction schema with entities+relations, verification schema with decisions).

### Key code details

- Entity alignment maps predicted entity IDs to gold vertexSet indices using 3-pass matching.
- Domain/range constraints are built from training data: for each relation P-code, store the set of (head_type, tail_type) pairs observed.
- Verification (Stage 2) processes candidates in batches of 10, asking the LLM to judge each candidate.

## 3. Baseline Results (for comparison)

```
Model: gemini-3-flash-preview (thinking_budget=0)
              Precision   Recall     F1    TP    FP    FN
Baseline           0.26     0.16   0.20    24    70   124
Two-Stage          0.36     0.22   0.27    32    56   116
```

**Key issue**: Recall is very low (0.16-0.22). With 35 relation types in the system prompt, the LLM may not attend to all of them equally, missing many valid relations.

## 4. Environment Setup

```bash
# Clone JacRED dataset
git clone https://github.com/YoumiMa/JacRED /tmp/JacRED

# Install dependencies
pip install google-genai openai

# Set API key
export GEMINI_API_KEY="<your-key>"
```

## 5. API Configuration

- **Model**: `gemini-3-flash-preview` (recommended) or `gemini-2.0-flash`
- **Structured Outputs**: `response_mime_type="application/json"` + `response_schema` dict
- **Temperature**: 0.2
- **ThinkingConfig**: `thinking_budget=0` for speed, `2048` for quality
- Configuration is in `llm_client.py` (the `MODEL` constant and `call_gemini()` function)

## 6. Task: Implement Relation-Split Multi-Pass Extraction

### Goal

Split the 35 relation types into 5 semantic groups and extract each group in a dedicated LLM pass, so the model can focus on fewer relation types at a time.

### Design

1. **Split relations into 5 groups**:

   - **Group 1 (Biographical)**: P569, P570, P19, P20, P27, P69, P108, P937
     - Birth date, death date, birthplace, deathplace, nationality, alma mater, employer, work location
   - **Group 2 (Family/Social)**: P40, P3373, P26
     - Child, sibling, spouse
   - **Group 3 (Geographic/Administrative)**: P131, P276, P1376, P36, P361, P527
     - Located in administrative entity, location, capital of, capital, part of, has part
   - **Group 4 (Organizational)**: P463, P6, P112, P127, P137, P1830, P121, P166
     - Member of, head of government, founded by, owned by, operator, owner of, item operated, award received
   - **Group 5 (Creative/Media)**: P170, P175, P123, P1441, P400, P674, P155, P156, P1344, P710
     - Creator, performer, publisher, present in work, platform, characters, follows, followed by, participant of, participant

2. **For each group**: One LLM call with a system prompt listing ONLY that group's relations and their Japanese definitions.

3. **Union all results** across groups (entities and triples).

4. **Apply domain/range constraint filtering** to remove invalid type combinations.

5. Optionally apply Stage 2 verification if precision needs improvement.

### Implementation Details

- **Add `RELATION_GROUPS` dict in `prompts.py`**:
  ```python
  RELATION_GROUPS = {
      "biographical": ["P569", "P570", "P19", "P20", "P27", "P69", "P108", "P937"],
      "family_social": ["P40", "P3373", "P26"],
      "geographic": ["P131", "P276", "P1376", "P36", "P361", "P527"],
      "organizational": ["P463", "P6", "P112", "P127", "P137", "P1830", "P121", "P166"],
      "creative_media": ["P170", "P175", "P123", "P1441", "P400", "P674", "P155", "P156", "P1344", "P710"],
  }
  ```

- **Add `build_group_system_prompt(group_name, group_pcodes, rel_info)` in `prompts.py`**:
  - Same structure as `build_system_prompt()` but only includes the relation types for the given group
  - Add a focused instruction: e.g., for biographical group: `"以下の人物に関する基本情報の関係のみを抽出してください。"`

- **Add `build_group_extraction_prompt(doc_text, few_shot_text, few_shot_output, group_pcodes)` in `prompts.py`**:
  - Similar to `build_extraction_prompt()` but filters few-shot output to only include relations from the current group
  - If no few-shot relations match the group, show the few-shot entities without relations as context

- **Add `run_relation_split()` function in `extraction.py`**:
  - Iterates over each group in `RELATION_GROUPS`
  - For each group: builds group-specific system prompt and extraction prompt, calls `call_gemini()`
  - Collects all entities and triples from all passes
  - Deduplicates entities (merge by normalized name)
  - Applies domain/range constraint filtering
  - Returns entities (union), final triples, and stats dict

- **Update `run_experiment.py`** to add the third condition

- **Entity merging across passes**:
  - Each pass will independently assign entity IDs (e0, e1, ...)
  - After collecting all passes, merge entities with the same normalized name
  - Reassign entity IDs in the merged triple list to use consistent IDs

### Expected Improvement

- **Better Recall**: Each pass focuses on fewer relation types, so the LLM's attention is concentrated and it is more likely to find relations of each type.
- **Precision**: Should be comparable to baseline since each pass uses the same structured output approach, plus constraint filtering.
- **Cost**: 5x baseline (one pass per group). Comparable to Majority Voting.

### Evaluation

- Same P/R/F1 computation on the same 10 dev documents
- Report per-document results and aggregate metrics
- Also report per-group extraction counts
- Compare: Baseline vs Two-Stage vs Relation-Split

### Output Format

The final comparison table should look like:
```
              Precision   Recall     F1    TP    FP    FN
Baseline           ...      ...    ...   ...   ...   ...
Two-Stage          ...      ...    ...   ...   ...   ...
RelSplit           ...      ...    ...   ...   ...   ...
```
