# automated_sales_enablement/agents/comparison_agent.py

from autogen import AssistantAgent

def comparison_agent(config_list):
    return AssistantAgent(
        name="ComparisonAgent",
        llm_config={"config_list": config_list},
        system_message="""Compare contract features with all historical releases.
        Output: JSON with released, planned, missing features."""
    )