def build_sales_context(contract_df, release_df, comparison, risk_data):
    """
    Builds a clear, explicit, sales-friendly context for the chatbot.
    This context is the SINGLE source of truth for chat answers.
    """

    lines = []

    # ---------------- CONTRACT COMMITMENTS ----------------
    if not contract_df.empty:
        lines.append("CUSTOMER CONTRACT COMMITMENTS:")
        for _, row in contract_df.iterrows():
            lines.append(
                f"- Feature {row['feature_id']} ({row['feature_name']}) "
                f"is committed in the contract with {row['priority']} priority."
            )

    # ---------------- RELEASE INFORMATION ----------------
    if not release_df.empty:
        lines.append("\nPRODUCT RELEASE INFORMATION:")
        for _, row in release_df.iterrows():
            lines.append(
                f"- Feature {row['feature_id']} ({row['feature_name']}) "
                f"has a release status of '{row['status']}'."
            )

    # ---------------- COMPUTED DELIVERY & RISK (KEY PART) ----------------
    summary = comparison.get("summary_table")

    if summary is not None and not summary.empty:
        lines.append("\nCURRENT DELIVERY STATUS AND RISK ASSESSMENT:")

        for _, row in summary.iterrows():
            feature_id = row["feature_id"]
            feature_name = row["feature_name"]
            status = row["status"]
            priority = row.get("priority", "Unknown")
            risk_level = row.get("risk_level", "NONE")
            risk_reason = row.get("risk_reason", "")

            if status.lower() in ["released", "done", "completed"]:
                lines.append(
                    f"- Feature {feature_id} ({feature_name}) HAS BEEN DELIVERED "
                    f"and is currently available to the customer."
                )

            elif status.lower() == "planned":
                lines.append(
                    f"- Feature {feature_id} ({feature_name}) is NOT YET DELIVERED "
                    f"but is planned for a future release. "
                    f"Priority: {priority}. Risk Level: {risk_level}."
                )

            else:  # Missing / Not Released
                lines.append(
                    f"- Feature {feature_id} ({feature_name}) has NOT BEEN DELIVERED yet. "
                    f"It is a {priority}-priority commitment. "
                    f"Risk Level: {risk_level}. {risk_reason}"
                )

    return "\n".join(lines)
