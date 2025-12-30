SQL = {
    "single_name": """
select document from langchain_pg_embedding where document ~ '^[A-Za-z0-9]+$';
        """,
    "find_plus_combination": """
select document from langchain_pg_embedding where document like '%+%';
        """,
    "find_fixed_combination": """
select document from langchain_pg_embedding where document like 'Fixed%';
        """,
}
