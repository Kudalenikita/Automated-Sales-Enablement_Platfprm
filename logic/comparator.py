# automated_sales_enablement/logic/comparator.py
# Survives ANY data state: empty, missing columns, NaN, no releases, etc.

import pandas as pd

def compare_features_agent(agent, contract_df: pd.DataFrame, release_df: pd.DataFrame) -> dict:
    """
    Safely compares contract features against release status.
    100% KeyError-proof.
    """
    # Default empty result structure
    empty_df = pd.DataFrame(columns=['feature_id', 'feature_name', 'description', 'priority', 'status'])

    # If no contract data at all
    if contract_df is None or contract_df.empty:
        return {
            "summary_table": empty_df,
            "released": empty_df,
            "planned": empty_df,
            "missing": empty_df,
        }

    # Ensure 'feature_id' exists
    if 'feature_id' not in contract_df.columns:
        return {
            "summary_table": empty_df,
            "released": empty_df,
            "planned": empty_df,
            "missing": empty_df,
        }

    # Build base summary from contract
    cols = ['feature_id']
    optional_cols = ['feature_name', 'description', 'priority']
    for col in optional_cols:
        if col in contract_df.columns:
            cols.append(col)

    summary_df = contract_df[cols].drop_duplicates(subset='feature_id').reset_index(drop=True)

    # Default: everything is Missing
    summary_df['status'] = 'Missing'

    # Only process releases if they have data and required columns
    if (
        release_df is not None
        and not release_df.empty
        and 'feature_id' in release_df.columns
        and 'status' in release_df.columns
    ):
        # Clean release data
        rel = release_df[['feature_id', 'status']].copy()

        # Drop rows where status is NaN or empty string
        rel = rel.dropna(subset=['status'])
        rel = rel[rel['status'].astype(str).str.strip() != '']

        # Convert to lowercase for comparison
        rel['status_clean'] = rel['status'].astype(str).str.strip().str.lower()

        # Remove 'nan' strings
        rel = rel[rel['status_clean'] != 'nan']

        if not rel.empty:
            # Determine status for each feature_id
            def get_feature_status(group_df):
                statuses = group_df['status_clean'].tolist()
                if any(s in ['released', 'done', 'completed', 'yes', 'true'] for s in statuses):
                    return 'Released'
                if 'planned' in statuses:
                    return 'Planned'
                return 'Missing'

            # Group and apply
            status_map = rel.groupby('feature_id').apply(get_feature_status).reset_index(name='release_status')

            # Merge into summary
            summary_df = summary_df.merge(status_map, on='feature_id', how='left')
            summary_df['status'] = summary_df['release_status'].fillna(summary_df['status'])
            summary_df.drop(columns=['release_status'], inplace=True, errors='ignore')

    # Final clean up
    summary_df['status'] = summary_df['status'].fillna('Missing')

    # Categorize
    released = summary_df[summary_df['status'] == 'Released'].reset_index(drop=True)
    planned = summary_df[summary_df['status'] == 'Planned'].reset_index(drop=True)
    missing = summary_df[summary_df['status'] == 'Missing'].reset_index(drop=True)

    summary_df = summary_df.reset_index(drop=True)

    return {
        "summary_table": summary_df,
        "released": released,
        "planned": planned,
        "missing": missing,
    }