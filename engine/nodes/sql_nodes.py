import json
from typing import Any, List, Literal

from langgraph.graph import END
from langgraph.types import Command, Send

from engine.states.risk_states import OverallState
from engine.states.sql_states import SqlState
from engine.tools.sql_constants import SQL
from engine.tools.sql_tools import (
    normalise_plus_combination,
    normalise_single_names,
    run_select_query,
    serialise_data_to_json,
)
from persistence.repositories import write_combinations_cache, write_single_names_cache


def select_single_names(state: SqlState) -> Command[Literal["warm_cache"]]:
    query = SQL["single_name"]
    rows = run_select_query(query)
    normalised_names = [normalise_single_names(name[0]) for name in rows]
    return Command(
        goto="warm_cache", update={"single_medicine_names": normalised_names}
    )


def select_combinations(state: SqlState) -> Command[Literal["warm_cache"]]:
    rows = run_select_query(SQL["find_plus_combination"])
    another_rows = run_select_query(SQL["find_fixed_combination"])

    normalised_plus = [normalise_plus_combination(r[0]) for r in rows]
    normalised_fixed = [name[0] for name in another_rows]

    return Command(
        goto="warm_cache",
        update={
            "combination_fixed_names": normalised_fixed,
            "combination_plus_names": normalised_plus,
        },
    )


def warm_cache(state: SqlState) -> Command[Literal[END]]:
    write_single_names_cache("banneddrugs", state["single_medicine_names"])
    write_combinations_cache("plus_combinations", state["combination_plus_names"])
    write_combinations_cache("fixed_dose", state["combination_fixed_names"])
    return Command(goto=END)


def middle_man(state: SqlState):
    return state
