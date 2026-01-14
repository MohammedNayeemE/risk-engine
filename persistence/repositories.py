import json
import os
import re
from threading import Lock
from typing import Any

import psycopg
from dotenv import load_dotenv
from langchain_postgres import PGVector
from pydantic import SerializeAsAny
from tavily import TavilyClient

from engine.etl.ingest import get_pgvector_store
from engine.states.risk_states import MedicineItem, MedicineList
from persistence.session import redis_client

load_dotenv()


def get_image_hash(
    image_hash: str, model: type[MedicineList] = MedicineList
) -> MedicineList | None:
    data = redis_client.get(image_hash)
    if data is None:
        return None

    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            return model.model_validate(parsed)
        except json.JSONDecodeError:
            return None
    return None


def set_image_hash(key: str, value: MedicineList) -> None:

    serialised = value.model_dump_json()
    redis_client.set(key, serialised)
    return None


def write_single_names_cache(key: str, value: list[str]) -> None:
    redis_client.set(key, json.dumps(value))


def write_combinations_cache(key: str, value: list[Any]) -> None:
    redis_client.set(key, json.dumps(value))


def get_single_names_cache(key: str) -> list[str]:
    data = redis_client.get(key)
    return json.loads(data) if data else []


def get_combinations_cache(key: str) -> list[Any]:
    data = redis_client.get(key)
    return json.loads(data) if data else []


def normalise_drug_names(medicine: MedicineItem) -> str:
    """
    Arg : MedicineItem
    normalises the names of the medicines by removing spaces and
    removing any content inside parenthesis and concatenating the numbers
    """
    if not isinstance(medicine, MedicineItem):
        return ""
    print(type(medicine))
    if not medicine or not medicine.name:
        return ""

    name: str = medicine.name
    name = name.lower()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[^a-z0-9\s]", "", name)
    return name.strip()


def normalise_drug_name(name: str) -> str:
    """
    Arg : name : class <'str'>
    normalises the name of the medicines by removing spaces and
    removing any content inside parenthesis and concatenating the numbers
    """
    name = name.lower()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[^a-z0-9\s]", "", name)
    return name.strip()


_lock = Lock()
_pg_vector_instance: dict[str, PGVector] = {}


def get_vector_db(collection_name: str) -> PGVector:
    with _lock:
        if collection_name not in _pg_vector_instance:
            _pg_vector_instance[collection_name] = get_pgvector_store(
                collection_name=collection_name
            )
        return _pg_vector_instance[collection_name]


def retrieve_from_sql(medicines: MedicineList) -> dict[str, dict]:
    """
    Returns:
        {
            "Augmentin": {...db data...},
            "PanD": {...db data...}
        }
    """

    print("here I am")

    if not medicines or not medicines.medicinelist:
        return {}

    conninfo = (
        f"host={os.getenv('PGHOST')} "
        f"port={os.getenv('PGPORT', 5432)} "
        f"user={os.getenv('PGUSER')} "
        f"password={os.getenv('PGPASSWORD')} "
        f"dbname={os.getenv('PGDATABASE')}"
    )

    exact_query = """
        SELECT medicine_name, data
        FROM new_medicines
        WHERE LOWER(medicine_name) = LOWER(%s)
        LIMIT 1;
    """
    fuzzy_query = """
        SELECT medicine_name, data
        FROM new_medicines
        WHERE medicine_name ILIKE %s
        ORDER BY 
            CASE WHEN medicine_name ILIKE %s THEN 0 ELSE 1 END,
            LENGTH(medicine_name)
        LIMIT 1;
    """

    results: dict[str, dict] = {}

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            for med in medicines.medicinelist:
                if not med.name:
                    continue

                name = med.name.strip()

                cur.execute(exact_query, (name,))
                row = cur.fetchone()

                if not row:
                    cur.execute(fuzzy_query, (f"%{name}%", name))
                    row = cur.fetchone()

                if row:
                    db_medicine_name, data = row
                    results[name] = {"db_name": db_medicine_name, "data": data}

    print("from sql", results)

    return {name: item["data"] for name, item in results.items()}


def search_medical_web(
    query: str, max_results: int = 5, include_raw_content: bool = False
) -> dict:
    """
    Search trusted medical domains using Tavily API.

    This tool searches only reputable medical and health information sources
    to ensure accurate and reliable information about medicines, treatments,
    and health conditions.

    Args:
        query: The search query string (e.g., "paracetamol side effects")
        max_results: Maximum number of results to return (default: 5)
        include_raw_content: Whether to include full page content (default: False)

    Returns:
        dict: A dictionary containing:
            - query: The original search query
            - results: List of search results with title, url, content, and score
            - answer: AI-generated summary answer (if available)
            - error: Error message if the search failed

    Example:
        >>> results = search_medical_web("aspirin dosage for adults")
        >>> print(results['answer'])
        >>> for result in results['results']:
        ...     print(f"{result['title']}: {result['url']}")
    """
    # List of trusted medical domains
    trusted_medical_domains = [
        "nih.gov",  # National Institutes of Health
        "cdc.gov",  # Centers for Disease Control
        "fda.gov",  # Food and Drug Administration
        "who.int",  # World Health Organization
        "mayoclinic.org",  # Mayo Clinic
        "drugs.com",  # Drugs.com
        "medlineplus.gov",  # MedlinePlus
        "ncbi.nlm.nih.gov",  # National Center for Biotechnology Information
        "webmd.com",  # WebMD
        "healthline.com",  # Healthline
        "rxlist.com",  # RxList
        "medscape.com",  # Medscape
        "bmj.com",  # British Medical Journal
        "nejm.org",  # New England Journal of Medicine
        "cochrane.org",  # Cochrane Library
        "pharmacytimes.com",  # Pharmacy Times
        "medicinenet.com",  # MedicineNet
    ]

    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {
                "query": query,
                "results": [],
                "answer": None,
                "error": "TAVILY_API_KEY not found in environment variables",
            }

        client = TavilyClient(api_key=api_key)

        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_domains=trusted_medical_domains,
            include_raw_content=include_raw_content,
            include_answer=True,
        )

        # Format results
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "raw_content": (
                        result.get("raw_content") if include_raw_content else None
                    ),
                }
            )

        return {
            "query": query,
            "results": formatted_results,
            "answer": response.get("answer"),
            "error": None,
        }

    except Exception as e:
        return {
            "query": query,
            "results": [],
            "answer": None,
            "error": f"Search failed: {str(e)}",
        }
