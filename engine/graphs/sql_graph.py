from langgraph.graph import END, START, StateGraph

from engine.nodes.sql_nodes import (
    middle_man,
    select_combinations,
    select_single_names,
    warm_cache,
)
from engine.states.sql_states import SqlState

builder = StateGraph(SqlState)

builder.add_node("middle_man", middle_man)
builder.add_node("select_single_names", select_single_names)
builder.add_node("select_combinations", select_combinations)
builder.add_node("warm_cache", warm_cache)

builder.add_edge(START, "middle_man")
builder.add_edge("middle_man", "select_combinations")
builder.add_edge("middle_man", "select_single_names")
builder.add_edge("select_combinations", "warm_cache")
builder.add_edge("select_single_names", "warm_cache")
builder.add_edge("warm_cache", END)

sql_graph = builder.compile()
