from operator import add
from typing import Annotated, List, TypedDict


class SqlState(TypedDict):
    single_medicine_names: Annotated[List[str], add]
    combination_plus_names: Annotated[List[str], add]
    combination_fixed_names: Annotated[List[str], add]
