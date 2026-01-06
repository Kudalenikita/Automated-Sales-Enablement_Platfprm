# automated_sales_enablement/logic/risk_engine.py

import asyncio
import json
import pandas as pd
from autogen_agentchat.agents import AssistantAgent

def risk_analysis_agent(agent: AssistantAgent, comparison_result: dict) -> dict:
    """
    Returns structured risk data:
    {
        "HIGH": count,
        "MEDIUM": count,
        "LOW": count,
        "NONE": count,
        "details": {feature_id: (risk_level, reason)},
        "summary_table": enhanced DataFrame with risk column
    }
    """
    summary_table = comparison_result["summary_table"]  # Should be pandas DataFrame
    
    if not isinstance(summary_table, pd.DataFrame):
        summary_table = pd.DataFrame(summary_table)

    # Apply deterministic risk logic (fallback if LLM fails)
    def assign_risk(row):
        priority = str(row.get("priority", "")).strip().lower()
        status = str(row.get("status", "")).strip()

        if priority == "high" and status in ["Missing", "Not Released"]:
            return "HIGH", "High-priority feature not yet released – escalation required"
        elif priority == "high" and status == "Planned":
            return "MEDIUM", "High-priority feature on roadmap but not live"
        elif status in ["Missing", "Not Released"]:
            return "MEDIUM", "Feature missing from current releases"
        elif status == "Planned":
            return "LOW", "Feature scheduled for future release"
        else:
            return "NONE", "Feature fully released and available"

    summary_table[["risk_level", "risk_reason"]] = summary_table.apply(
        lambda row: pd.Series(assign_risk(row)), axis=1
    )

    # Count risks
    risk_counts = summary_table["risk_level"].value_counts().to_dict()
    
    details = {
        row["feature_id"]: (row["risk_level"], row["risk_reason"])
        for _, row in summary_table.iterrows()
    }

    return {
        "HIGH": risk_counts.get("HIGH", 0),
        "MEDIUM": risk_counts.get("MEDIUM", 0),
        "LOW": risk_counts.get("LOW", 0),
        "NONE": risk_counts.get("NONE", 0),
        "details": details,
        "summary_table": summary_table
    }