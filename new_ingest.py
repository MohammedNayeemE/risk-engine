import json
import re
from pathlib import Path
from typing import Any, Dict, List

import psycopg
from dotenv import load_dotenv

load_dotenv()


def save_to_postgres(kv: dict):

    conninfo = (
        "host=localhost "
        "port=5432 "
        "user=postgres "
        "password=@genworx.ai "
        "dbname=rag_db"
    )
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            for key, value in kv.items():
                name = key.replace("medicine:", "")

                cur.execute(
                    """
                    INSERT INTO medicines (name, data)
                    VALUES (%s, %s)
                    ON CONFLICT (name)
                    DO UPDATE SET
                        data = EXCLUDED.data,
                        updated_at = now()
                    """,
                    (name, json.dumps(value)),
                )

        conn.commit()


# -----------------------------
# Utils
# -----------------------------


def normalize_name(name: str) -> str:
    return (
        name.lower()
        .replace("**", "")
        .replace("=", "")
        .replace("(", "")
        .replace(")", "")
        .strip()
    )


# -----------------------------
# Markdown table handling
# -----------------------------

TABLE_PATTERN = re.compile(
    r"(\|.+?\|\n\|[-| ]+\|\n(?:\|.*?\|\n?)+)",
    re.MULTILINE,
)


def extract_tables(text: str) -> List[str]:
    return TABLE_PATTERN.findall(text)


def parse_md_table(table: str) -> List[Dict[str, str]]:
    lines = [l.strip() for l in table.splitlines() if l.strip()]

    if len(lines) < 3:
        return []

    headers = [h.strip() for h in lines[0].split("|")[1:-1]]

    rows = []
    for row in lines[2:]:
        cells = [c.strip() for c in row.split("|")[1:-1]]
        rows.append(dict(zip(headers, cells)))

    return rows


def extract_section(block: str, title: str) -> str | None:
    pattern = rf"### \*\*{re.escape(title)}\*\*\n([\s\S]*?)(?=\n###|\Z)"
    match = re.search(pattern, block)
    return match.group(1).strip() if match else None


# -----------------------------
# Medicine block parsing
# -----------------------------


def parse_medicine(block: str) -> Dict[str, Any]:
    name_match = re.search(r"# \*\*(.+?)\*\*", block)
    if not name_match:
        return {}

    raw_name = name_match.group(1)
    name = normalize_name(raw_name)

    tables = extract_tables(block)
    parsed_tables = [parse_md_table(t) for t in tables if parse_md_table(t)]

    return {
        "meta": {
            "name": name,
            "raw_name": raw_name,
        },
        "therapeutic_action": extract_section(block, "Therapeutic action"),
        "indications": extract_section(block, "Indications"),
        "forms_and_strengths": extract_section(block, "Forms and strengths"),
        "dosage_text": extract_section(block, "Dosage"),
        "contraindications": extract_section(
            block, "Contra-indications, adverse effects, precautions"
        ),
        "storage": extract_section(block, "Storage"),
        "tables": parsed_tables,
    }


# -----------------------------
# Split full MD into medicines
# -----------------------------


def split_medicines(text: str) -> List[str]:
    pattern = r"\n# \*\*(.+?)\*\*\n"
    matches = list(re.finditer(pattern, text))

    blocks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append(text[start:end])

    return blocks


# -----------------------------
# Main ingestion function
# -----------------------------


def ingest_md_to_dict(md_path: str) -> Dict[str, Dict[str, Any]]:
    text = Path(md_path).read_text(encoding="utf-8")
    medicine_blocks = split_medicines(text)

    kv_store: Dict[str, Dict[str, Any]] = {}

    for block in medicine_blocks:
        data = parse_medicine(block)
        if data and data["meta"]["name"]:
            key = f"medicine:{data['meta']['name']}"
            kv_store[key] = data

    return kv_store


if __name__ == "__main__":
    md_file = "./4llm-output.md"

    kv_data = ingest_md_to_dict(md_file)
    save_to_postgres(kv_data)

    print("DONE............")
