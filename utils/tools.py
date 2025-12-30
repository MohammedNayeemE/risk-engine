import json
import re
from threading import Lock
from typing import Any

from langchain_postgres import PGVector
from pydantic import SerializeAsAny

from ingest import get_pgvector_store
from utils.redis_client import redis_client
from utils.states import MedicineItem, MedicineList


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
