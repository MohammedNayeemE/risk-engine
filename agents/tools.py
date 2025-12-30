import json
import os
import re
from typing import Any, List

import psycopg
from dotenv import load_dotenv

load_dotenv()


def normalise_single_names(name: str) -> str:
    name = name.lower()
    return name.strip()


def normalise_plus_combination(name: str) -> Any:
    name = name.lower()
    name = re.sub(
        r"^\s*\**\s*fixed dose combination of\s*",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(
        r"\s*(for human use.*|s\.o\..*|g\.s\.r\..*|dispesible tablets.*)$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    name = re.sub(r"\*+", "", name)
    parts = re.split(r"\s*\+\s*", name)

    parts = [p.strip().lower() for p in parts]
    return tuple(sorted(parts))


def serialise_data_to_json(data: Any) -> str:
    return json.dumps(data)


def run_select_query(sql: str) -> List[Any]:

    assert sql.lower().strip().startswith("select")

    conninfo = (
        "host=localhost "
        "port=5432 "
        "user=postgres "
        "password=@genworx.ai "
        "dbname=rag_db"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as curr:
            curr.execute(sql)
            rows = curr.fetchall()

    return rows
