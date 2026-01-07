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


AGENT_MAP = {
    "identity": "identity_agent",
    "strength": "strength_agent",
    "dosage": "dosage_agent",
    "age": "age_agent",
    "regulatory": "regulatory_agent",
}
