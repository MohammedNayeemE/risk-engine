import json
import os
import re
from datetime import datetime
from threading import Lock
from typing import Any, Optional

import psycopg
from dotenv import load_dotenv
from langchain_postgres import PGVector
from pydantic import SerializeAsAny
from tavily import TavilyClient

from engine.etl.ingest import get_pgvector_store
from engine.states.risk_states import MedicineItem, MedicineList
from persistence.models import APIKey, APIKeyWithSecret, Client
from persistence.session import redis_client
from utils.api_key import generate_api_key_with_hash

load_dotenv()


# ---------------------------------------------------------------------------
# Thread Management Functions
# ---------------------------------------------------------------------------


def save_thread(
    thread_id: str,
    user_name: str,
    status: str = "active",
    metadata: Optional[dict] = None,
) -> None:
    """
    Save thread metadata to the database.

    Args:
        thread_id: Unique thread identifier
        user_name: Name of the user associated with the thread
        status: Thread status (active, completed, failed, etc.)
        metadata: Additional metadata as JSON
    """
    conninfo = (
        f"host={os.getenv('PGHOST')} "
        f"port={os.getenv('PGPORT', 5432)} "
        f"user={os.getenv('PGUSER')} "
        f"password={os.getenv('PGPASSWORD')} "
        f"dbname={os.getenv('PGDATABASE')}"
    )
    metadata_json = json.dumps(metadata or {})

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO threads (thread_id, user_name, status, metadata)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (thread_id) 
                DO UPDATE SET 
                    status = EXCLUDED.status,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (thread_id, user_name, status, metadata_json),
            )
            conn.commit()


def get_thread(thread_id: str) -> Optional[dict]:
    """
    Retrieve thread metadata from the database.

    Args:
        thread_id: Thread identifier to look up

    Returns:
        Dictionary with thread data or None if not found
    """
    conninfo = (
        f"host={os.getenv('PGHOST')} "
        f"port={os.getenv('PGPORT', 5432)} "
        f"user={os.getenv('PGUSER')} "
        f"password={os.getenv('PGPASSWORD')} "
        f"dbname={os.getenv('PGDATABASE')}"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT thread_id, user_name, status, created_at, updated_at, metadata
                FROM threads
                WHERE thread_id = %s
                """,
                (thread_id,),
            )
            row = cur.fetchone()

            if not row:
                return None

            return {
                "thread_id": row[0],
                "user_name": row[1],
                "status": row[2],
                "created_at": row[3],
                "updated_at": row[4],
                "metadata": row[5],
            }


def update_thread_status(thread_id: str, status: str) -> None:
    """
    Update the status of an existing thread.

    Args:
        thread_id: Thread identifier
        status: New status value
    """
    conninfo = (
        f"host={os.getenv('PGHOST')} "
        f"port={os.getenv('PGPORT', 5432)} "
        f"user={os.getenv('PGUSER')} "
        f"password={os.getenv('PGPASSWORD')} "
        f"dbname={os.getenv('PGDATABASE')}"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE threads
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = %s
                """,
                (status, thread_id),
            )
            conn.commit()


def get_user_threads(user_name: str, limit: int = 10) -> list[dict]:
    """
    Get all threads for a specific user.

    Args:
        user_name: User to get threads for
        limit: Maximum number of threads to return

    Returns:
        List of thread dictionaries
    """
    conninfo = (
        f"host={os.getenv('PGHOST')} "
        f"port={os.getenv('PGPORT', 5432)} "
        f"user={os.getenv('PGUSER')} "
        f"password={os.getenv('PGPASSWORD')} "
        f"dbname={os.getenv('PGDATABASE')}"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT thread_id, user_name, status, created_at, updated_at, metadata
                FROM threads
                WHERE user_name = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (user_name, limit),
            )
            rows = cur.fetchall()

            return [
                {
                    "thread_id": row[0],
                    "user_name": row[1],
                    "status": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                    "metadata": row[5],
                }
                for row in rows
            ]


# ---------------------------------------------------------------------------
# Existing Functions
# ---------------------------------------------------------------------------


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
    # print(type(medicine))
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

    # print("here I am")

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

    # print("from sql", results)

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


# ---------------------------------------------------------------------------
# Client Management Functions
# ---------------------------------------------------------------------------


def create_client(
    company_name: str, email: str, domain: str
) -> tuple[Client, APIKeyWithSecret]:
    """
    Create a new client and generate their first API key.

    Args:
        company_name: Name of the client company
        email: Client's email address
        domain: Frontend domain for CORS validation

    Returns:
        Tuple of (Client, APIKeyWithSecret) - the created client and their API key
    """
    conninfo = (
        f"host={os.getenv('PGHOST')} "
        f"port={os.getenv('PGPORT', 5432)} "
        f"user={os.getenv('PGUSER')} "
        f"password={os.getenv('PGPASSWORD')} "
        f"dbname={os.getenv('PGDATABASE')}"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            # Create client
            cur.execute(
                """
                INSERT INTO clients (company_name, email, domain, is_active)
                VALUES (%s, %s, %s, %s)
                RETURNING id, company_name, email, domain, is_active, created_at, updated_at
                """,
                (company_name, email, domain, True),
            )
            client_row = cur.fetchone()
            client = Client(
                id=client_row[0],
                company_name=client_row[1],
                email=client_row[2],
                domain=client_row[3],
                is_active=client_row[4],
                created_at=client_row[5],
                updated_at=client_row[6],
            )

            # Generate API key
            api_key, key_hash, key_prefix = generate_api_key_with_hash()

            # Store API key
            cur.execute(
                """
                INSERT INTO api_keys (client_id, key_hash, key_prefix, name, is_active)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, client_id, key_hash, key_prefix, name, is_active, 
                          last_used_at, created_at, expires_at
                """,
                (client.id, key_hash, key_prefix, "Default API Key", True),
            )
            key_row = cur.fetchone()
            api_key_obj = APIKeyWithSecret(
                id=key_row[0],
                client_id=key_row[1],
                key_hash=key_row[2],
                key_prefix=key_row[3],
                name=key_row[4],
                is_active=key_row[5],
                last_used_at=key_row[6],
                created_at=key_row[7],
                expires_at=key_row[8],
                api_key=api_key,
            )

            conn.commit()
            return client, api_key_obj


def get_client_by_id(client_id: int) -> Optional[Client]:
    """
    Retrieve a client by their ID.

    Args:
        client_id: The client's ID

    Returns:
        Client object or None if not found
    """
    conninfo = (
        f"host={os.getenv('PGHOST')} "
        f"port={os.getenv('PGPORT', 5432)} "
        f"user={os.getenv('PGUSER')} "
        f"password={os.getenv('PGPASSWORD')} "
        f"dbname={os.getenv('PGDATABASE')}"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, company_name, email, domain, is_active, created_at, updated_at
                FROM clients
                WHERE id = %s
                """,
                (client_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            return Client(
                id=row[0],
                company_name=row[1],
                email=row[2],
                domain=row[3],
                is_active=row[4],
                created_at=row[5],
                updated_at=row[6],
            )


def get_client_by_email(email: str) -> Optional[Client]:
    """
    Retrieve a client by their email.

    Args:
        email: The client's email address

    Returns:
        Client object or None if not found
    """
    conninfo = (
        f"host={os.getenv('PGHOST')} "
        f"port={os.getenv('PGPORT', 5432)} "
        f"user={os.getenv('PGUSER')} "
        f"password={os.getenv('PGPASSWORD')} "
        f"dbname={os.getenv('PGDATABASE')}"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, company_name, email, domain, is_active, created_at, updated_at
                FROM clients
                WHERE email = %s
                """,
                (email,),
            )
            row = cur.fetchone()
            if not row:
                return None

            return Client(
                id=row[0],
                company_name=row[1],
                email=row[2],
                domain=row[3],
                is_active=row[4],
                created_at=row[5],
                updated_at=row[6],
            )


def validate_api_key(api_key: str) -> Optional[tuple[Client, APIKey]]:
    """
    Validate an API key and return the associated client and key info.

    Args:
        api_key: The API key to validate

    Returns:
        Tuple of (Client, APIKey) if valid, None otherwise
    """
    from utils.api_key import hash_api_key

    key_hash = hash_api_key(api_key)
    conninfo = (
        f"host={os.getenv('PGHOST')} "
        f"port={os.getenv('PGPORT', 5432)} "
        f"user={os.getenv('PGUSER')} "
        f"password={os.getenv('PGPASSWORD')} "
        f"dbname={os.getenv('PGDATABASE')}"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            # Get API key and client
            cur.execute(
                """
                SELECT 
                    k.id, k.client_id, k.key_hash, k.key_prefix, k.name, k.is_active,
                    k.last_used_at, k.created_at, k.expires_at,
                    c.id, c.company_name, c.email, c.domain, c.is_active, 
                    c.created_at, c.updated_at
                FROM api_keys k
                JOIN clients c ON k.client_id = c.id
                WHERE k.key_hash = %s AND k.is_active = true AND c.is_active = true
                """,
                (key_hash,),
            )
            row = cur.fetchone()
            if not row:
                return None

            # Check if key is expired
            if row[8] and row[8] < datetime.now():
                return None

            api_key_obj = APIKey(
                id=row[0],
                client_id=row[1],
                key_hash=row[2],
                key_prefix=row[3],
                name=row[4],
                is_active=row[5],
                last_used_at=row[6],
                created_at=row[7],
                expires_at=row[8],
            )

            client = Client(
                id=row[9],
                company_name=row[10],
                email=row[11],
                domain=row[12],
                is_active=row[13],
                created_at=row[14],
                updated_at=row[15],
            )

            # Update last_used_at
            cur.execute(
                """
                UPDATE api_keys 
                SET last_used_at = CURRENT_TIMESTAMP 
                WHERE id = %s
                """,
                (api_key_obj.id,),
            )
            conn.commit()

            return client, api_key_obj


def create_api_key_for_client(
    client_id: int, name: str = "API Key"
) -> Optional[APIKeyWithSecret]:
    """
    Create a new API key for an existing client.

    Args:
        client_id: The client's ID
        name: Optional name for the API key

    Returns:
        APIKeyWithSecret object or None if client not found
    """
    # Check if client exists
    client = get_client_by_id(client_id)
    if not client:
        return None

    api_key, key_hash, key_prefix = generate_api_key_with_hash()
    conninfo = (
        f"host={os.getenv('PGHOST')} "
        f"port={os.getenv('PGPORT', 5432)} "
        f"user={os.getenv('PGUSER')} "
        f"password={os.getenv('PGPASSWORD')} "
        f"dbname={os.getenv('PGDATABASE')}"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO api_keys (client_id, key_hash, key_prefix, name, is_active)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, client_id, key_hash, key_prefix, name, is_active,
                          last_used_at, created_at, expires_at
                """,
                (client_id, key_hash, key_prefix, name, True),
            )
            row = cur.fetchone()
            conn.commit()

            return APIKeyWithSecret(
                id=row[0],
                client_id=row[1],
                key_hash=row[2],
                key_prefix=row[3],
                name=row[4],
                is_active=row[5],
                last_used_at=row[6],
                created_at=row[7],
                expires_at=row[8],
                api_key=api_key,
            )


def revoke_api_key(api_key_id: int) -> bool:
    """
    Revoke (deactivate) an API key.

    Args:
        api_key_id: The ID of the API key to revoke

    Returns:
        True if successful, False otherwise
    """
    conninfo = (
        f"host={os.getenv('PGHOST')} "
        f"port={os.getenv('PGPORT', 5432)} "
        f"user={os.getenv('PGUSER')} "
        f"password={os.getenv('PGPASSWORD')} "
        f"dbname={os.getenv('PGDATABASE')}"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE api_keys 
                SET is_active = false, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                RETURNING id
                """,
                (api_key_id,),
            )
            result = cur.fetchone()
            conn.commit()
            return result is not None


def list_client_api_keys(client_id: int) -> list[APIKey]:
    """
    List all API keys for a client.

    Args:
        client_id: The client's ID

    Returns:
        List of APIKey objects
    """
    conninfo = (
        f"host={os.getenv('PGHOST')} "
        f"port={os.getenv('PGPORT', 5432)} "
        f"user={os.getenv('PGUSER')} "
        f"password={os.getenv('PGPASSWORD')} "
        f"dbname={os.getenv('PGDATABASE')}"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, client_id, key_hash, key_prefix, name, is_active,
                       last_used_at, created_at, expires_at
                FROM api_keys
                WHERE client_id = %s
                ORDER BY created_at DESC
                """,
                (client_id,),
            )
            rows = cur.fetchall()

            return [
                APIKey(
                    id=row[0],
                    client_id=row[1],
                    key_hash=row[2],
                    key_prefix=row[3],
                    name=row[4],
                    is_active=row[5],
                    last_used_at=row[6],
                    created_at=row[7],
                    expires_at=row[8],
                )
                for row in rows
            ]
