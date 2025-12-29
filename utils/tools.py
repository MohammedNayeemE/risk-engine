import re

from langchain_postgres import PGVector

from ingest import get_pgvector_store
from utils.states import MedicineItem, MedicineList


def normalise_drug_names(medicine: MedicineItem) -> str:
    """
    Arg : MedicineItem
    normalises the names of the medicines by removing spaces and
    removing any content inside parenthesis and concatenating the numbers
    """
    if isinstance(medicine, str):
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


# TODO: use single instance of the postgres
def get_vector_db(collection_name: str) -> PGVector:
    return get_pgvector_store(collection_name=collection_name)
