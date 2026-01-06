# automated_sales_enablement/agents/risk_agent.py

from autogen import AssistantAgent

def risk_agent(config_list):
    return AssistantAgent(
        name="RiskAgent",
        llm_config={"config_list": config_list},
        system_message="""Analyze risk:
        - High priority not released: HIGH
        - Planned delayed: MEDIUM
        - Released: LOW/NONE
        Output: Risk levels per feature."""
    )