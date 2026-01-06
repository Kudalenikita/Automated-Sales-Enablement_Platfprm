# automated_sales_enablement/agents/ingestion_agent.py

from autogen import AssistantAgent

def ingestion_agent(config_list):
    return AssistantAgent(
        name="IngestionAgent",
        llm_config={"config_list": config_list},
        system_message="""You are the ingestion agent. Validate CSV schema, normalize text, chunk, embed, store in SQLite and Vector DB.
        For contracts: Store once per customer.
        For releases: Accumulate history."""
    )