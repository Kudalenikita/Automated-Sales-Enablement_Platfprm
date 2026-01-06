# automated_sales_enablement/app.py


import streamlit as st
import pandas as pd
import os
import sqlite3
import hashlib
import warnings
import logging
import asyncio
import pickle
import time
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


# AutoGen imports
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent

# Local imports
from db.db_utils import (
    init_db,
    store_contract_to_db,
    store_release_to_db,
    load_contracts_for_customer,
    load_all_releases_for_customer,
)
from rag.rag_engine import ingest_to_vector_db, query_vector_db
from logic.comparator import compare_features_agent
from logic.risk_engine import risk_analysis_agent
from logic.sales_insight import create_sales_insight_agent
from logic.sales_context import build_sales_context
from logic.pitch_deck import generate_pitch_deck_content_sync, build_pptx_from_content

from utils.utils import normalize_text, chunk_text

# Suppress noisy logs
warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Sales Enablement AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ── Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .pill-button {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        border-radius: 999px;
        padding: 4px;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.25);
    }
    
    .pill-button > button {
        background: #1e293b !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        font-size: 1.02rem !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.25s ease !important;
    }
    
    .pill-button > button:hover {
        background: #334155 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(124, 58, 237, 0.35) !important;
    }

    .exec-summary-container {
        background: #1e1b32;
        border: 1px solid #4b3f72;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin: 1rem 0;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.18);
        font-size: 0.93rem;
        line-height: 1.5;
        color: #e0e7ff;
    }

    .exec-summary-container strong {
        color: #c4b5fd;
    }

    .exec-summary-confidence {
        margin-top: 1rem;
        font-size: 0.8rem;
        color: #94a3b8;
        font-style: italic;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# Purple pill button styling (unchanged)
st.markdown("""
<style>
    .pill-button {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        border-radius: 999px;
        padding: 4px;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.25);
    }
    
    .pill-button > button {
        background: #1e293b !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        font-size: 1.02rem !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.25s ease !important;
    }
    
    .pill-button > button:hover {
        background: #334155 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(124, 58, 237, 0.35) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Authentication ====================
# Simple SQLite-based authentication with hashed passwords (SHA-256).
# Users are stored in data/users.db. Not production-grade (no rate limiting, etc.)
# but sufficient for internal sales tool.

def init_user_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/users.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, name TEXT)""")
    conn.commit()
    conn.close()

init_user_db()

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("<h2 style='text-align: center;'>🔐 Sales Login</h2>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Login", "Signup"])
        with tab1:
            with st.form("login"):
                user = st.text_input("Username")
                pw = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    conn = sqlite3.connect("data/users.db")
                    c = conn.cursor()
                    c.execute("SELECT password FROM users WHERE username=?", (user,))
                    res = c.fetchone()
                    conn.close()
                    if res and hash_password(pw) == res[0]:
                        st.session_state.logged_in = True
                        st.session_state.username = user
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        with tab2:
            with st.form("signup"):
                name = st.text_input("Full Name")
                new_user = st.text_input("New Username")
                new_pw = st.text_input("New Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Create Account"):
                    if new_pw != confirm:
                        st.error("Passwords don't match")
                    elif len(new_pw) < 6:
                        st.error("Password too short")
                    else:
                        conn = sqlite3.connect("data/users.db")
                        c = conn.cursor()
                        try:
                            c.execute("INSERT INTO users VALUES (?, ?, ?)",
                                      (new_user, hash_password(new_pw), name))
                            conn.commit()
                            st.success("Account created! Login now.")
                        except sqlite3.IntegrityError:
                            st.error("Username taken")
                        conn.close()
    st.stop()

# Header
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown(f"<small>Logged in as: <strong>{st.session_state.username}</strong></small>",
                unsafe_allow_html=True)
with col2:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

st.markdown(
    "<h1 style='text-align: center; background: linear-gradient(90deg, #a855f7, #ec4899, #10b981); "
    "-webkit-background-clip: text; -webkit-text-fill-color: transparent;'>"
    "Automated Sales Enablement</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #94a3b8;'>"
    "Close deals with confidence. AI that aligns promises with delivery.</p>",
    unsafe_allow_html=True
)

# ==================== Persistent State ====================
# Persists uploaded files metadata, chat history, and pitch deck state across reruns.
# Uses pickle for simplicity — ensure trusted environment only.

PERSISTENT_FILE = "data/persistent_state.pkl"

def load_persistent_state():
    if os.path.exists(PERSISTENT_FILE):
        try:
            with open(PERSISTENT_FILE, "rb") as f:
                state = pickle.load(f)
                for key in ["uploaded_contracts", "uploaded_releases", "chat_sessions", "current_chat_id", "pitch_deck_path", "pitch_generated", "download_time"]:
                    if key in state:
                        st.session_state[key] = state[key]
        except Exception as e:
            st.error(f"Error loading persistent state: {e}")

def save_persistent_state():
    state_to_save = {
        "uploaded_contracts": st.session_state.get("uploaded_contracts", []),
        "uploaded_releases": st.session_state.get("uploaded_releases", []),
        "chat_sessions": st.session_state.get("chat_sessions", {}),
        "current_chat_id": st.session_state.get("current_chat_id"),
        "pitch_deck_path": st.session_state.get("pitch_deck_path", None),
        "pitch_generated": st.session_state.get("pitch_generated", False),
        "download_time": st.session_state.get("download_time", None)
    }
    os.makedirs("data", exist_ok=True)
    with open(PERSISTENT_FILE, "wb") as f:
        pickle.dump(state_to_save, f)

load_persistent_state()

# Right after load_persistent_state()
defaults = {
    "chat_sessions": {},
    "current_chat_id": None,
    "pitch_deck_path": None,
    "pitch_generated": False,
    "download_time": None,
    "page": "Dashboard",
    "uploaded_contracts": [],
    "uploaded_releases": [],
    "selected_risk_level": None,

    # 🔔 upload notifications
    "contract_notice_time": None,
    "release_notice_time": None,
    "single_contract_warn_time": None,
}

for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

if "existing_data_notice_time" not in st.session_state:
    st.session_state.existing_data_notice_time = None

if "existing_data_toast_time" not in st.session_state:
    st.session_state.existing_data_toast_time = None

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown("### 🚀 Navigation")

    def nav_button(label, icon, page_name):
        is_active = st.session_state.page == page_name

        st.button(
            f"{icon}  {label}",
            use_container_width=True,
            type="secondary" if is_active else "tertiary",
            key=f"nav_{page_name}",
            on_click=lambda: st.session_state.update(page=page_name)
        )


    # Navigation buttons
    nav_button("Dashboard", "🏠", "Dashboard")
    nav_button("Upload Data", "📤", "Upload Data")
    nav_button("Chat / Sales Assistant", "💬", "Chat / Sales Assistant")

    if st.session_state.uploaded_contracts or st.session_state.uploaded_releases:
        nav_button("Uploaded Files", "📁", "Uploaded Files")


    st.markdown("---")
    st.markdown("### Status")
    if OPENAI_API_KEY:
        st.success("✅ OpenAI Connected")
    else:
        st.error("❌ Add OPENAI_API_KEY to .env")
    has_contract = len(st.session_state.uploaded_contracts) == 1
    has_releases = len(st.session_state.uploaded_releases) >= 1
    data_fully_loaded = has_contract and has_releases
    if data_fully_loaded:
        st.success("✅ Data Loaded")
        if st.button("🗑️ Clear All Data", type="primary"):
            keys_to_clear = ["chat_sessions", "current_chat_id", "pitch_deck_path", "pitch_generated", "download_time", "uploaded_contracts", "uploaded_releases", "page", "selected_risk_level"]
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            if os.path.exists("data/sales.db"):
                os.remove("data/sales.db")
            if os.path.exists("data/chroma"):
                try:
                    import shutil
                    shutil.rmtree("data/chroma")
                except Exception as e:
                    st.warning(f"Chroma cleanup issue: {e}")
            if os.path.exists(PERSISTENT_FILE):
                os.remove(PERSISTENT_FILE)
            st.rerun()
    else:
        st.info("No data uploaded yet")
        st.markdown("---")
# ==================== Executive Summary Function ====================
def build_executive_summary(customer_name: str, contract_df: pd.DataFrame, release_df: pd.DataFrame, risk_data: dict) -> str:
    if contract_df is None or contract_df.empty or release_df is None or release_df.empty:
        delivered_pct = 0
        high_risk = 0
        medium_risk = 0
    else:
        merged = contract_df.merge(release_df[['feature_id', 'status']], on='feature_id', how='left')
        total = len(merged)
        released = len(merged[merged['status'] == 'Released'])
        planned = len(merged[merged['status'] == 'Planned'])
        delivered_pct = int(((released + planned) / max(total, 1)) * 100)
        high_risk = risk_data.get("HIGH", 0) if risk_data else 0
        medium_risk = risk_data.get("MEDIUM", 0) if risk_data else 0
    
    summary = f"""**This quarter:**
• **{delivered_pct}%** of committed features delivered or in progress

**Risk Status:**
• High risk: **{high_risk}**
• Medium risk: **{medium_risk}**

**Account Outlook:**
• Well positioned for renewal
• Upsell opportunity identified

**Recommended Sales Actions:**
1. Reinforce delivered value
2. Position roadmap for upsell
3. Maintain proactive risk communication"""
    
    return summary.strip()

# ==================== Vector DB & RAG Setup ====================
# ChromaDB persistent client stored in data/chroma
# Uses OpenAI's text-embedding-3-small for cost-efficient embeddings
# All contract and release chunks are ingested with metadata for retrieval
# (Chunks created in upload processing via ingest_to_vector_db calls)
  
# ==================== Initialization ====================
if not OPENAI_API_KEY:
    st.stop()

os.makedirs("data", exist_ok=True)
init_db()

vector_client = chromadb.PersistentClient(path="data/chroma")
embedding_func = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)


embedding_func = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)







model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)

#==================== AutoGen Agents ====================
# Three lightweight agents:
# - ComparisonAgent: matches contract features vs released/planned features
# - RiskAgent: assigns risk levels (HIGH/MEDIUM/LOW/NONE) based on delivery gaps
# - PitchDeckAgent: generates structured pitch deck content (title, slides, speaker notes)

comparison_agent = AssistantAgent(name="ComparisonAgent", model_client=model_client)
risk_agent = AssistantAgent(name="RiskAgent", model_client=model_client)
pitch_agent = AssistantAgent(name="PitchDeckAgent", model_client=model_client)

def run_agent_sync(agent: AssistantAgent, task: str) -> str:
    resp = asyncio.run(agent.run(task=task))

    # AutoGen v0.7+ preferred output
    if hasattr(resp, "final_output") and resp.final_output:
        return str(resp.final_output).strip()

    # Safe fallback
    if hasattr(resp, "messages") and resp.messages:
        last = resp.messages[-1]
        if hasattr(last, "content"):
            return last.content.strip()
        return "No response generated."

def get_file_hash(file) -> str:
    file.seek(0)
    content = file.read()
    file.seek(0)
    return hashlib.sha256(content).hexdigest()

# ==================== Upload Data Page ====================
# Enforces strict rules:
#   • Exactly ONE customer contract CSV
#   • One or more release CSVs
# Duplicates detected via SHA-256 hash to prevent re-processing

if st.session_state.page == "Upload Data":
    # ---- Check if data already exists ----
    has_contract = len(st.session_state.uploaded_contracts) == 1
    has_releases = len(st.session_state.uploaded_releases) >= 1
    data_ready = has_contract and has_releases

    now = time.time()

    if data_ready and st.session_state.existing_data_toast_time is None:
        st.session_state.existing_data_toast_time = now
        save_persistent_state()

    if data_ready and st.session_state.existing_data_notice_time is None:
        st.session_state.existing_data_notice_time = now
        save_persistent_state()

        # Show message only for 10 seconds
        if now - st.session_state.existing_data_notice_time < 10:
            st.info("✅ Data is already loaded and ready!")
            st.write(
                "You can view analysis on the **Dashboard** or upload new files below "
                "(this will replace current data)."
            )
            if st.button("← Back to Dashboard"):
                st.session_state.page = "Dashboard"
                st.rerun()
        else:
            # Auto-hide after 10 seconds
            st.session_state.existing_data_notice_time = None
            save_persistent_state()
            st.rerun()

    st.markdown("### 📁 Upload Contract & Release Data")
    contract_file = st.file_uploader("Customer Contract CSV (exactly one required)", type=["csv"], key="contract_uploader")
    release_file = st.file_uploader("Product Release Notes CSV (at least one required)", type=["csv"], key="release_uploader")

    # ---- Notification placeholder (below upload section) ----
    notice_container = st.container()

    processed = False  # Flag to detect if any new upload happened

    if contract_file:
        file_hash = get_file_hash(contract_file)
        duplicate = any(
            name == contract_file.name and h == file_hash
            for name, h, _, _ in st.session_state.uploaded_contracts
        )
        if duplicate:
            st.session_state.single_contract_warn_time = time.time()
        else:
            if len(st.session_state.uploaded_contracts) >= 1:
                st.session_state.single_contract_warn_time = time.time()
            else:
                with st.spinner("Processing contract..."):
                    df = pd.read_csv(contract_file)
                    required = ["customer_name", "feature_id", "feature_name", "description", "priority"]
                    if not all(c in df.columns for c in required):
                        st.error(f"Missing columns. Need: {', '.join(required)}")
                    else:
                        for _, row in df.iterrows():
                            store_contract_to_db(row.to_dict())
                            text = normalize_text(
                                f"Contract: {row['feature_name']} — {row['description']} (Priority: {row['priority']})"
                            )
                            for chunk in chunk_text(text):
                                ingest_to_vector_db(
                                    vector_client, embedding_func, chunk,
                                    {"type": "contract", "customer_name": row["customer_name"], "feature_id": row["feature_id"]}
                                )
                        timestamp = datetime.now()
                        st.session_state.uploaded_contracts = [(contract_file.name, file_hash, timestamp, df.copy())]
                        st.success("Contract uploaded successfully!")
                        st.session_state.contract_notice_time = time.time()
                        save_persistent_state()
                        processed = True  # Mark as processed

    if release_file:
        file_hash = get_file_hash(release_file)
        duplicate = any(
            name == release_file.name and h == file_hash
            for name, h, _, _ in st.session_state.uploaded_releases
        )
        if duplicate:
            st.session_state.release_notice_time = time.time()
        else:
            with st.spinner("Processing releases..."):
                df = pd.read_csv(release_file)
                required = ["customer_name", "feature_id", "feature_name", "status"]
                if not all(c in df.columns for c in required):
                    st.error(f"Missing columns. Need: {', '.join(required)}")
                else:
                    for _, row in df.iterrows():
                        store_release_to_db(row.to_dict())
                        text = normalize_text(
                            f"Release: {row['feature_name']} — Status: {row['status']}"
                        )
                        for chunk in chunk_text(text):
                            ingest_to_vector_db(
                                vector_client, embedding_func, chunk,
                                {"type": "release", "customer_name": row["customer_name"], "feature_id": row["feature_id"]}
                            )
                    timestamp = datetime.now()
                    st.session_state.uploaded_releases.append(
                        (release_file.name, file_hash, timestamp, df.copy())
                    )
                    st.success("Release file uploaded successfully!")
                    st.session_state.release_notice_time = time.time()
                    save_persistent_state()
                    processed = True  # Mark as processed

    # After any processing, check if now ready and redirect
    if processed:
        has_contract = len(st.session_state.uploaded_contracts) == 1
        has_releases = len(st.session_state.uploaded_releases) >= 1
        if has_contract and has_releases:
            st.success("✅ Data uploaded successfully! Redirecting to Dashboard...")
            st.session_state.page = "Dashboard"
            save_persistent_state()
            st.rerun()
            st.stop()

    with notice_container:
        # ---- Contract loaded notification (10s) ----
        if st.session_state.contract_notice_time:
            if now - st.session_state.contract_notice_time < 10:
                st.toast("✅ Contract loaded", icon="📄")
            else:
                st.session_state.contract_notice_time = None
                save_persistent_state()
                st.rerun()

        # ---- Release files loaded notification (10s) ----
        if st.session_state.release_notice_time:
            if now - st.session_state.release_notice_time < 10:
                st.toast(
                    f"✅ {len(st.session_state.uploaded_releases)} release file(s) loaded"
                )
            else:
                st.session_state.release_notice_time = None
                save_persistent_state()
                st.rerun()

        # ---- Single contract warning (10s) ----
        if st.session_state.single_contract_warn_time:
            if now - st.session_state.single_contract_warn_time < 10:
                st.toast(
                    "Only one customer contract allowed. Clear data to upload another.",
                    icon="⚠️"
                )
            else:
                st.session_state.single_contract_warn_time = None
                save_persistent_state()
                st.rerun()

        if st.session_state.existing_data_toast_time:
            if now - st.session_state.existing_data_toast_time < 10:
                contract_name = st.session_state.uploaded_contracts[0][0]
                release_count = len(st.session_state.uploaded_releases)

                st.toast(
                f"""
                **📄 Active Contract:** `{contract_name}`  
                **📦 Release Files Loaded:** `{release_count}`
                """
            )
            else:
                st.session_state.existing_data_toast_time = None
                save_persistent_state()
                st.rerun()

elif st.session_state.page == "Uploaded Files":
    st.markdown("""
    <p style='text-align: center; color: #cbd5e1; font-size: 0.95rem; margin-top: 0;'>
        Your contract and release data — ready for AI analysis ✨
    </p>
    """, unsafe_allow_html=True)
    if not st.session_state.uploaded_contracts and not st.session_state.uploaded_releases:
        st.info("No files uploaded yet. Visit **Upload Data** to get started!")
    else:
        if st.session_state.uploaded_contracts:
            name, _, ts, df = st.session_state.uploaded_contracts[0]
            with st.container(border=True):
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #a855f7, #9333ea);
                            padding: 12px; border-radius: 10px; color: white; margin-bottom: 8px;'>
                    <h4 style='margin:0; font-size:1.1rem;'>📄 Customer Contract</h4>
                    <p style='margin:4px 0 0 0; font-size:0.85rem; opacity:0.9;'>
                        Active & Processed ✓
                    </p>
                </div>
                """, unsafe_allow_html=True)
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown(f"<small style='color:#d8b4fe;'>**File:** <code style='background:#1e293b; padding:2px 6px; border-radius:4px; font-size:0.8rem;'>{name}</code></small>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<small style='color:#d8b4fe;'>📅 {ts.strftime('%b %d, %Y ⋅ %H:%M')}</small>", unsafe_allow_html=True)
                st.markdown(f"<small style='color:#d8b4fe;'>📊 {len(df)} rows • {df['customer_name'].nunique()} customer(s)</small>", unsafe_allow_html=True)
                with st.expander("👁️ View Contract Data", expanded=False):
                    st.dataframe(df, use_container_width=True, hide_index=True)

        if st.session_state.uploaded_releases:
            st.markdown("#### 📑 Product Release Notes")
            st.markdown("<small style='color:#e9d5ff;'>All features tracked and ready</small>", unsafe_allow_html=True)
            cols = st.columns(min(3, len(st.session_state.uploaded_releases)))
            purple_shades = ["#c084fc", "#a855f7", "#9333ea", "#7c3aed", "#6d28d9"]
            for idx, (name, _, ts, df) in enumerate(st.session_state.uploaded_releases):
                color = purple_shades[idx % len(purple_shades)]
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, {color}, #7c3aed);
                                    padding: 10px; border-radius: 10px; color: white; margin-bottom: 8px;'>
                            <h5 style='margin:0; font-size:1rem;'>{name}</h5>
                            <p style='margin:3px 0 0 0; font-size:0.8rem; opacity:0.9;'>
                                Ready ✨
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"<small style='color:#d8b4fe;'>📅 {ts.strftime('%b %d, %Y ⋅ %H:%M')}</small>", unsafe_allow_html=True)
                        st.markdown(f"<small style='color:#d8b4fe;'>✨ {len(df)} features</small>", unsafe_allow_html=True)
                        with st.expander("👁️ View Data", expanded=False):
                            st.dataframe(df, use_container_width=True, hide_index=True)

else:
    customers = set()
    conn = sqlite3.connect("data/sales.db")
    cur = conn.cursor()
    try:
        cur.execute("SELECT DISTINCT customer_name FROM customers")
        for row in cur.fetchall():
            customers.add(row[0])
    except:
        pass
    conn.close()

    for _, _, _, df in st.session_state.uploaded_contracts:
        if "customer_name" in df.columns:
            customers.update(df["customer_name"].unique())

    customers = sorted(list(customers))
    if not customers:
        st.warning("No customers found. Please upload contract data first.")
        st.stop()

    customer = st.selectbox("👤 Select Customer", customers)
    contract_df = load_contracts_for_customer(customer)
    release_df = load_all_releases_for_customer(customer)

    # ---- SHARED COMPUTED TRUTH (Dashboard + Chat) ----
    comparison = compare_features_agent(comparison_agent, contract_df, release_df)
    risk_data = risk_analysis_agent(risk_agent, comparison)
    
# ==================== Dashboard Page ====================
# Main analysis view for the selected customer.
# Shared computed truth:
#   • comparison = compare_features_agent(...) → feature matching table
#   • risk_data = risk_analysis_agent(...) → risk counts + summary_table
# These results are computed once and reused in both Dashboard and Chat pages.


# Features:
#   • Interactive risk cards with click-to-filter (HIGH/MEDIUM/LOW/NONE)
#   • Filtered feature status table
#   • One-click AI pitch deck generation (structured content → PPTX)
#   • Deterministic executive summary with delivery % and risk highlights
#   • Strict download button: only enabled after successful generation
#   • 10-second success toast after download

    if st.session_state.page == "Dashboard":
        if "selected_risk_level" not in st.session_state:
            st.session_state.selected_risk_level = None

        st.markdown("### Risk Overview")
        st.markdown("<small style='color:#94a3b8;'>Click a risk level to instantly view the affected features.</small>", unsafe_allow_html=True)

        high_count = risk_data.get("HIGH", 0)
        medium_count = risk_data.get("MEDIUM", 0)
        low_count = risk_data.get("LOW", 0)
        none_count = risk_data.get("NONE", 0)

        st.markdown(f"""
        <style>
            .compact-risk-card {{
                box-shadow: 0 3px 8px rgba(0,0,0,0.1);
                transition: all 0.2s ease;
                border-radius: 10px;
                padding: 14px 10px;
                text-align: center;
                cursor: pointer;
                height: 90px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }}
            .compact-risk-card:hover {{
                transform: translateY(-4px);
                box-shadow: 0 6px 16px rgba(0,0,0,0.15);
            }}
            .compact-risk-card.selected {{
                border: 3px solid #a855f7;
                transform: scale(1.03);
            }}
        </style>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0;">
            <div class="compact-risk-card {'selected' if st.session_state.selected_risk_level == 'HIGH' else ''}" 
                 onclick="parent.document.querySelector('[key=\\"risk_high_click\\"] button').click()">
                <div style="font-size: 2.2rem; font-weight: bold; color: #dc2626; line-height: 1;">{high_count}</div>
                <div style="font-size: 0.95rem; color: #991b1b; margin-top: 4px;">🔴 High Risk</div>
            </div>
            <div class="compact-risk-card {'selected' if st.session_state.selected_risk_level == 'MEDIUM' else ''}" 
                 onclick="parent.document.querySelector('[key=\\"risk_medium_click\\"] button').click()">
                <div style="font-size: 2.2rem; font-weight: bold; color: #d97706; line-height: 1;">{medium_count}</div>
                <div style="font-size: 0.95rem; color: #92400e; margin-top: 4px;">🟡 Medium Risk</div>
            </div>
            <div class="compact-risk-card {'selected' if st.session_state.selected_risk_level == 'LOW' else ''}" 
                 onclick="parent.document.querySelector('[key=\\"risk_low_click\\"] button').click()">
                <div style="font-size: 2.2rem; font-weight: bold; color: #059669; line-height: 1;">{low_count}</div>
                <div style="font-size: 0.95rem; color: #047857; margin-top: 4px;">🟢 Low Risk</div>
            </div>
            <div class="compact-risk-card {'selected' if st.session_state.selected_risk_level == 'NONE' else ''}" 
                 onclick="parent.document.querySelector('[key=\\"risk_none_click\\"] button').click()">
                <div style="font-size: 2.2rem; font-weight: bold; color: #059669; line-height: 1;">{none_count}</div>
                <div style="font-size: 0.95rem; color: #047857; margin-top: 4px;">✅ No Risk</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("HIGH", key="risk_high_click", use_container_width=True):
                st.session_state.selected_risk_level = "HIGH"
                st.rerun()
        with col2:
            if st.button("MEDIUM", key="risk_medium_click", use_container_width=True):
                st.session_state.selected_risk_level = "MEDIUM"
                st.rerun()
        with col3:
            if st.button("LOW", key="risk_low_click", use_container_width=True):
                st.session_state.selected_risk_level = "LOW"
                st.rerun()
        with col4:
            if st.button("NONE", key="risk_none_click", use_container_width=True):
                st.session_state.selected_risk_level = "NONE"
                st.rerun()

        st.markdown("""
        <style>
            div[data-testid="column"] button {
                visibility: hidden !important;
                height: 0 !important;
                padding: 0 !important;
                margin: 0 !important;
                min-height: 0 !important;
            }
        </style>
        """, unsafe_allow_html=True)

        if st.session_state.selected_risk_level is not None:
            if st.button("← Back to Overview", type="secondary", use_container_width=True):
                st.session_state.selected_risk_level = None
                st.rerun()

        summary_table = risk_data["summary_table"].copy()
        possible_cols = ["risk", "risk_level", "Risk Level", "Risk", "risk_category", "level"]
        risk_col = next((col for col in possible_cols if col in summary_table.columns), None)

        if risk_col is None:
            st.warning("Risk column not found — showing all features.")
            filtered_table = summary_table
        else:
            summary_table[risk_col] = summary_table[risk_col].astype(str).str.upper().str.strip()
            if st.session_state.selected_risk_level is None:
                filtered_table = summary_table
            else:
                filtered_table = summary_table[
                    summary_table[risk_col] == st.session_state.selected_risk_level
                ].reset_index(drop=True)
            


        st.markdown("### Feature Status Table")
        st.dataframe(filtered_table, use_container_width=True, hide_index=False)
        
# ==================== Pitch Deck Generation ====================
# Flow:
# 1. generate_pitch_deck_content_sync() → structured dict via LLM
# 2. build_pptx_from_content() → uses python-pptx to create downloadable .pptx
# Executive summary is derived deterministically (no LLM) for consistency.

        st.markdown("### ✨ AI-Powered Sales Pitch Deck")
        st.markdown("<small style='color:#94a3b8;'>Generates a complete AI-powered pitch deck along with a concise executive summary .</small>", unsafe_allow_html=True)

        col_g1, col_g2, col_g3 = st.columns([1, 3, 1])
        with col_g2:
            
            if st.button(
                "✨ Generate AI Sales Pitch Deck",
                type="primary",
                use_container_width=True,
                key="generate_pitch_deck_btn"
            ):
                with st.spinner("Crafting professional pitch deck and summary... This may take 20–40 seconds."):
                    try:
                        content = generate_pitch_deck_content_sync(
                            pitch_agent,
                            customer,
                            vector_client,
                            embedding_func,
                            comparison,
                            risk_data
                        )
                        if not isinstance(content, dict):
                            raise TypeError("Generated content is not a dictionary")

                        file_path = build_pptx_from_content(content, customer)
                        st.session_state.pitch_deck_path = file_path
                        st.session_state.pitch_generated = True
                        
                        st.session_state.executive_summary = build_executive_summary(
                            customer, contract_df, release_df, risk_data
                        )
                        st.session_state.executive_summary_visible = True
                        
                        save_persistent_state()
                        st.success("🎉 Pitch deck generated successfully!")
                        st.toast(f"Pitch deck ready for {customer}!", icon="✅")
                        st.rerun()

                    except Exception as e:
                        st.error("Failed to generate pitch deck.")
                        st.exception(e)
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.pitch_generated and st.session_state.get("pitch_deck_path") and os.path.exists(st.session_state.pitch_deck_path):
            st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
            
            if "executive_summary" in st.session_state and st.session_state.executive_summary_visible:
                col_header, col_actions = st.columns([8, 2])
                with col_header:
                    st.markdown(
        "<div style='font-size:0.9rem;  font-weight:600;'>📋 Sales Readiness Summary</div>",
        unsafe_allow_html=True
    )
                with col_actions:
                    col_close, col_copy = st.columns([1, 1])
                    with col_close:
                        if st.button("✖", key="close_summary_btn", type="secondary"):
                            st.session_state.executive_summary_visible = False
                            st.rerun()
                    with col_copy:
                        if st.button("📋 Copy", key="copy_summary_btn"):
                            full_text = f"{st.session_state.executive_summary}\n\nAI confidence: High (based on current delivery and risk signals)"
                            st.markdown(f"""
                            <script>
                            navigator.clipboard.writeText(`{full_text}`);
                            </script>
                            """, unsafe_allow_html=True)
                            st.toast("Summary copied to clipboard!", icon="✅")

                st.markdown(f"""
                <div class="exec-summary-container">
                    <div class="exec-summary-bullets">
                        {st.session_state.executive_summary}
                    </div>
                    <div class="exec-summary-confidence">
                        AI confidence: High (based on current delivery and risk signals)
                    </div>
                </div>
                """, unsafe_allow_html=True)

            col_d1, col_d2, col_d3 = st.columns([1, 3, 1])
            with col_d2:
                
                with open(st.session_state.pitch_deck_path, "rb") as f:
                    downloaded = st.download_button(
                        label="📥 Download Pitch Deck (PPTX)",
                        data=f,
                        file_name=os.path.basename(st.session_state.pitch_deck_path),
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True,
                        key="download_pitch_final"
                    )
                st.markdown("</div>", unsafe_allow_html=True)

                if downloaded:
                    st.session_state.download_time = time.time()
                    save_persistent_state()
                    st.session_state.executive_summary_visible = False
                    st.rerun()

        if st.session_state.get("download_time"):
            elapsed = time.time() - st.session_state.download_time
            if elapsed < 10:
                st.toast("Downloaded successfully", icon="✅")
            else:
                st.session_state.download_time = None
                save_persistent_state()
                st.rerun()
                
# ==================== Chat / Sales Assistant ====================
# Multi-chat support with persistent history
# Each user message triggers:
#   • build_sales_context() → concise RAG-retrieved + computed context
#   • create_sales_insight_agent() → temporary agent with system prompt
#   • Single LLM call for fast, focused answers

    elif st.session_state.page == "Chat / Sales Assistant":
        st.markdown("### 💬 AI Sales Assistant")
        st.caption("Ask about risks, objections, pitch, timing — direct & confident answers only.")
        st.markdown("""
        <style>
        .chat-container {
            background: #020617;
            border: 1px solid #1e293b;
            border-radius: 16px;
            
            max-width: 820px;
            margin-left: auto;
            margin-right: auto;
        }
        .user-msg {
            background: linear-gradient(135deg, #7c3aed, #9333ea);
            color: white;
            padding: 10px 14px;
            border-radius: 14px 14px 4px 14px;
            max-width: 75%;
            margin-left: auto;
            margin-bottom: 10px;
        }
        .assistant-msg {
            background: #020617;
            border: 1px solid #1e293b;
            color: #e5e7eb;
            padding: 12px 14px;
            border-radius: 14px 14px 14px 4px;
            max-width: 75%;
            margin-right: auto;
            margin-bottom: 10px;
        }
        .user-msg, .assistant-msg {
            animation: fadeIn 0.25s ease-in;
        }
        .active-chat {
            background: linear-gradient(135deg, #7c3aed, #9333ea) !important;
            border: 2px solid #6366f1;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """, unsafe_allow_html=True)




        # --------- TWO COLUMN LAYOUT ---------
        left, right = st.columns([0.55, 2.45]) 

        # --------- LEFT: YOUR CHATS ---------
        with left:
            st.markdown("#### 🗂️ Your Chats")

            if st.button("➕ New Chat", use_container_width=True):
                chat_id = f"chat_{int(time.time())}"
                st.session_state.chat_sessions[chat_id] = {
                    "title": "New Chat",
                    "messages": []
                }
                st.session_state.current_chat_id = chat_id
                save_persistent_state()
                st.rerun()

            st.divider()

            if st.session_state.chat_sessions:
                for chat_id, chat in st.session_state.chat_sessions.items():
                    is_active = (st.session_state.current_chat_id == chat_id)

                    col1, col2 = st.columns([0.88, 0.30])

                    with col1:
                        is_active = (st.session_state.current_chat_id == chat_id)

                        button_style = "active-chat" if is_active else ""

                        st.markdown(
                            f"""
                            <div class="{button_style}">
                            """,
                            unsafe_allow_html=True
                        )

                        if st.button(chat["title"], key=f"select_{chat_id}", use_container_width=True):
                            st.session_state.current_chat_id = chat_id
                            save_persistent_state()
                            st.rerun()

                    with col2:
                        with st.popover(""):
                            if st.button(
                                "🗑️ Delete",
                                key=f"confirm_delete_{chat_id}",
                                type="primary"
                            ):
                                del st.session_state.chat_sessions[chat_id]
                                if st.session_state.current_chat_id == chat_id:
                                    st.session_state.current_chat_id = None
                                save_persistent_state()
                                st.rerun()


            else:
                st.info("No chats yet — click ➕ New Chat to start!")


        # --------- RIGHT: CHAT WINDOW ---------
        with right:
            if not st.session_state.current_chat_id:
                st.info("Start a new chat or select an existing one.")
                st.stop()

            chat = st.session_state.chat_sessions[st.session_state.current_chat_id]
            messages = chat["messages"]

            if messages:
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)

                for idx,msg in enumerate(messages):
                    role = msg["role"]
                    content = msg["content"]

                    if role == "user":
                        st.markdown(
                            f'<div class="user-msg">{content}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="assistant-msg">
                                <pre style="
                                    white-space: pre-wrap;
                                    word-wrap: break-word;
                                    margin: 0;
                                    font-family: inherit;
                                ">{content}</pre>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="
                    background:#020617;
                    border:1px dashed #1e293b;
                    border-radius:16px;
                    padding:40px;
                    text-align:center;
                    color:#94a3b8;
                ">
                    💬 Start the conversation by asking a question below
                </div>
                """, unsafe_allow_html=True)

            prompt = st.chat_input("Ask about risks, objections, timelines…")

            if prompt:
                # Add user message
                messages.append({"role": "user", "content": prompt})

                # Auto-name chat
                if chat["title"] == "New Chat":
                    chat["title"] = f"Q: {prompt[:18]}..."

                # Generate assistant response
                with st.spinner("Thinking..."):
                    sales_context = build_sales_context(
                        contract_df, release_df, comparison, risk_data
                    )

                    if not sales_context.strip():
                        response = "Not found in provided contract or release data."
                    else:
                        agent = create_sales_insight_agent()
                        task_prompt = f"""
            Customer: {customer}
            Question: {prompt}

            Context:
            {sales_context}
            """
                        response = run_agent_sync(agent, task_prompt)

                # Add assistant message ONCE
                messages.append({"role": "assistant", "content": response})

                save_persistent_state()
                st.rerun()


# Footer
st.markdown(
    "<hr><p style='text-align:center; color:#64748b;'>"
    "Enterprise Sales AI • Pure RAG + Agentic • Powered by GPT-4o-mini • 2026</p>",
    unsafe_allow_html=True
)
