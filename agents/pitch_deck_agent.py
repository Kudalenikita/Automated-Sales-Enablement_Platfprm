# automated_sales_enablement/agents/pitch_deck_agent.py

from autogen import AssistantAgent

def pitch_deck_agent(config_list):
    return AssistantAgent(
        name="PitchDeckAgent",
        llm_config={"config_list": config_list},
        system_message="""Generate content for 6-7 slide pitch deck:
        1. Overview
        2. Commitments
        3. Delivered
        4. Roadmap
        5. Risk & Mitigation
        6. Value
        7. Next Steps"""
    )