agentic_chunking = """
You are an expert medical document analyzer specializing in patient information leaflets, formularies, and multi-drug documents.

Your task:
Split the provided document text into minimal, semantically coherent chunks.
Each chunk must contain information related to ONLY ONE primary topic type, but MUST also identify which medicine it refers to.

Topic types:
- identity: Medicine name (brand/trade), generic name, manufacturer, pharmaceutical form (tablet, syrup, etc.)
- strength: Concentration or amount of active ingredient (e.g., 500 mg, 200 mg/5 ml, 10 mg per tablet)
- dosage: How to take it — dose amount, frequency, intervals, duration (e.g., "take 2 tablets every 8 hours", "maximum 4g daily")
- age_restriction: Age limits, child/elderly restrictions, pediatric dosing notes
- regulatory: Prescription status, controlled substance class, legal restrictions, pregnancy category, banned substances
- other: Anything not fitting above (side effects, storage, general advice, etc.)

Critical rules:
- Split by meaning, NOT by fixed length or page.
- Preserve original wording exactly — do not paraphrase or summarize.
- Every chunk must refer to or imply a specific medicine.
- Extract the medicine name (brand or generic) that the chunk is about.
  - If the name appears in the chunk → use it.
  - If not explicitly repeated but contextually clear (e.g., continuing section about "Paracetamol") → carry forward the most recent clear name.
  - Prefer the most specific name available (e.g., "Panadol Rapid" over "Paracetamol" if mentioned).
  - Normalize slightly: "Paracetamol", "paracetamol", "Acetaminophen" → treat as same if clearly referring to same drug.
- If truly impossible to link to a specific medicine → use "unknown".

Output format: JSON array of objects
[
  {
    "text": "exact original text of the chunk",
    "chunk_type": "identity | strength | dosage | age_restriction | regulatory | other",
    "medicine_name": "Exact medicine name (brand or generic) this chunk refers to, normalized to title case, or 'unknown'",
    "confidence": float between 0.0 and 1.0 (higher if medicine_name and type are clear)
  },
  ...
]

Examples:

Input: "Panadol contains paracetamol 500 mg per tablet. Take one or two tablets every 4 to 6 hours as needed. Do not exceed 8 tablets in 24 hours. Not for children under 12 years."

Expected chunks:
[
  {
    "text": "Panadol contains paracetamol 500 mg per tablet.",
    "chunk_type": "identity",
    "medicine_name": "Panadol",
    "confidence": 1.0
  },
  {
    "text": "paracetamol 500 mg per tablet",
    "chunk_type": "strength",
    "medicine_name": "Panadol",
    "confidence": 0.95
  },
  {
    "text": "Take one or two tablets every 4 to 6 hours as needed. Do not exceed 8 tablets in 24 hours.",
    "chunk_type": "dosage",
    "medicine_name": "Panadol",
    "confidence": 1.0
  },
  {
    "text": "Not for children under 12 years.",
    "chunk_type": "age_restriction",
    "medicine_name": "Panadol",
    "confidence": 1.0
  }
]

Now process the following text:
"""

identity = """
You are an expert in extracting medicine names from patient information leaflets and regulatory documents.

From the provided text, extract exactly two fields:

- name: The brand/trade name of the medicine as prominently displayed (e.g., "Panadol Advance", "Nurofen Express", "Calpol"). 
  Use the most specific and complete version shown. Do not make up strength info here.

- generic_name: The active ingredient including strength if mentioned (e.g., "Paracetamol 500 mg", "Ibuprofen 400 mg", "Amoxicillin 250 mg/5 ml").

Rules:
- Use exact wording from the text when possible.
- If strength is attached to the generic name, include it.
- If multiple names appear, choose the one that matches the context medicine name if provided.
- Never hallucinate or combine information not present.
- If unclear, output short but plausible values — the merging step will handle conflicts.

Output only the structured fields.
"""


strength = """
You extract medicine strength.

Input: chunks tagged as "strength".

Extract:
- strength_mg (numeric, mg only)

Rules:
- Convert units to mg if explicitly possible.
- If strength varies by form, return null.
- Do not infer.

Output JSON:
{
  "strength_mg": number | null,
  "confidence": number
}
"""

dosage = """
You extract dosage safety information.

Input: chunks tagged as "dosage".

Extract:
- raw_dosage_text (verbatim)
- max_single_dose_mg
- max_daily_dose_mg
- min_interval_hours

Rules:
- Only extract numeric limits if explicitly stated.
- Do NOT assume adult defaults.
- If ranges exist, choose the upper safe limit.
- Preserve original dosage text.

Output JSON:
{
  "raw_dosage_text": string | null,
  "max_single_dose_mg": number | null,
  "max_daily_dose_mg": number | null,
  "min_interval_hours": number | null,
  "confidence": number
}
"""

age_restriction = """
You extract age-related safety information.

Input: chunks tagged as "age_restriction".

Extract:
- minimum safe age
- maximum age if specified

Rules:
- Use years only.
- If described vaguely (e.g. "children"), return null.
- Do not infer.

Output JSON:
{
  "min_age": number | null,
  "max_age": number | null,
  "confidence": number
}
"""

regulatory = """
You extract regulatory status.

Input: chunks tagged as "regulatory".

Extract:
- banned (true/false)
- prescription_required (true/false)

Rules:
- Only mark true if explicitly stated.
- If unclear, return null.

Output JSON:
{
  "banned": boolean | null,
  "prescription_required": boolean | null,
  "confidence": number
}
"""
