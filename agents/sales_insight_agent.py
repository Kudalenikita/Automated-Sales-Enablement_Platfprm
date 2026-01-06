# logic/sales_insight.py
# Fixes model_info error by explicitly providing model_info dict

import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent


def create_sales_insight_agent() -> AssistantAgent:
    system_message = """
You are a Senior Sales Enablement AI. Answer ONLY the exact question asked.

STRICT RULES (never break):
- Bullet points only
- Maximum 5 bullets
- No greetings, introductions, explanations, or closings
- Never summarize features or contracts
- Use ONLY the provided RAG context
- If no relevant risk → reply exactly: "No contract or release risk identified for this customer."

When asked about risk(s):
- Format: [Risk description] if [feature] not enabled
- Include business impact
- End with confident sales talking point

Example:
- Fraud exposure if AI fraud detection not enabled
- Significant compliance delays without automated reporting
- Cash-flow blind spots if real-time payment tracking not active
- Limited scalability without open API integration
- Recommend prioritizing high-priority features in next quarter

Other questions: short, confident, positive bullets only.

Answer now using only the context.
"""

    # EXPLICIT model_info FIX – this resolves the dict/property bug
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "gpt",
            "structured_output": True
        }
    )

    return AssistantAgent(
        name="SalesInsightAgent",
        system_message=system_message,
        model_client=model_client
    )