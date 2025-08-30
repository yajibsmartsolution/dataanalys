# app.py
# ⚡ INJECTION SUBSTATION NERVOUS SYSTEM v2.0
# Top 0.1% Grade: Real-time monitoring, intelligent alerts, role-based control
# Author: Yaquba + AI Architect (Grid OS Level)

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import base64
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

# -------------------------------
# CONFIGURATION & CONSTANTS
# -------------------------------
DATA_PATH = "HOURLY LOAD SHEET.xlsx"
SUBMISSION_FILE = "submitted_data.csv"
ALERT_TIMEOUT = 17  # seconds
AUDIO_ALERT = True  # enable beep on alert

# Role credentials (in prod: use DB or OAuth)
ADMIN_CREDENTIALS = {"admin": "power@2025", "englead": "grid#master"}
OPERATOR_PREFIX = "OPR"  # enforce format: OPR001, OPR002...

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------

def inject_audio():
    """Inject hidden audio element for alerts"""
    audio_bytes = base64.b64encode(b"""
    UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=
    """).decode()  # Silent beep placeholder (use real WAV in prod)
    st.markdown(f"""
    <audio id="alert-sound" src="data:audio/wav;base64,{audio_bytes}" preload="auto"></audio>
    <script>
    function playAlert() {{
        var sound = document.getElementById('alert-sound');
        if (sound) sound.play().catch(e => console.log('Audio play failed:', e));
    }}
    </script>
    """, unsafe_allow_html=True)

def play_alert():
    st.write('<script>playAlert()</script>', unsafe_allow_html=True)

def get_remote_ip() -> str:
    """Get user session IP (for multi-user tracking)"""
    try:
        ctx = get_script_run_ctx()
        if ctx and ctx.request:
            return ctx.request.remote_ip
    except Exception: pass
    return "unknown"

def log_event(level, message, operator=None):
    ip = get_remote_ip()
    log_entry = pd.DataFrame([{
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Level": level,
        "Operator": operator or "System",
        "IP": ip,
        "Message": message
    }])
    try:
        log = pd.read_csv("audit_log.csv")
        log = pd.concat([log, log_entry], ignore_index=True)
    except: log = log_entry
    log.to_csv("audit_log.csv", index=False)

# -------------------------------
# DATA LOADING & CACHING
# -------------------------------
@st.cache_data(ttl=300)
def load_data(path):
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    df["FEEDER ID"] = range(len(df))  # unique ID for CRUD
    return df.copy()

@st.cache_data
def get_substations():
    df = load_data(DATA_PATH)
    return sorted(df["33/11KV INJECTION SUBSTATION"].dropna().unique())

# -------------------------------
# SESSION STATE INIT
# -------------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None
    st.session_state.start_time = None
    st.session_state.alert_fired = False

# -------------------------------
# AUTHENTICATION LAYER
# -------------------------------
def login():
    st.subheader("🔐 Grid Access Portal")
    role = st.radio("Login As", ["Operator", "Admin"], key="login_role")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password") if role == "Admin" else None

    if st.button("🔓 Connect to Grid"):
        if role == "Operator":
            if username.startswith(OPERATOR_PREFIX) and len(username) >= 5:
                st.session_state.logged_in = True
                st.session_state.role = "Operator"
                st.session_state.username = username
                st.session_state.start_time = time.time()
                log_event("INFO", f"Operator login", username)
                st.rerun()
            else:
                st.error(f"Operator ID must start with '{OPERATOR_PREFIX}' (e.g., OPR001)")
        elif role == "Admin":
            if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
                st.session_state.logged_in = True
                st.session_state.role = "Admin"
                st.session_state.username = username
                log_event("CRITICAL", "Admin login", username)
                st.rerun()
            else:
                st.error("Invalid admin credentials")

# -------------------------------
# NAVIGATION
# -------------------------------
def render_sidebar():
    st.sidebar.title(f"⚡ {st.session_state.role} Console")
    st.sidebar.write(f"👤 {st.session_state.username}")
    page = st.sidebar.radio("Navigate", ["📊 Data Entry", "📈 Live Dashboard", "⚙️ Feeder Manager"])
    if st.sidebar.button("🚪 Logout"):
        log_event("INFO", "User logged out", st.session_state.username)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    return page

# -------------------------------
# INACTIVITY ALERT SYSTEM
# -------------------------------
def check_inactivity():
    if st.session_state.start_time:
        elapsed = time.time() - st.session_state.start_time
        if elapsed > ALERT_TIMEOUT and not st.session_state.alert_fired:
            st.session_state.alert_fired = True
            st.error(f"⏰ **URGENT**: No input within {ALERT_TIMEOUT}s! Please enter data.")
            if AUDIO_ALERT:
                play_alert()
            log_event("WARNING", f"Inactivity timeout: {ALERT_TIMEOUT}s", st.session_state.username)

# -------------------------------
# FEEDER MANAGEMENT (Admin Only)
# -------------------------------
def manage_feeders():
    st.subheader("⚙️ Feeder Configuration")
    df = load_data(DATA_PATH)
    subs = get_substations()
    selected_sub = st.selectbox("Select Substation", subs, key="admin_sub")

    sub_feeders = df[df["33/11KV INJECTION SUBSTATION"] == selected_sub]

    with st.expander("🗂️ Current Feeders"):
        st.dataframe(sub_feeders[[
            "FEEDER NAME", "PEAK LOAD EVER", "BAND", "ENERGY READING (PREVIOUS)"
        ]])

    with st.form("add_feeder"):
        st.write("➕ Add New Feeder")
        new_name = st.text_input("Feeder Name")
        new_peak = st.number_input("Peak Load Ever (MW)", min_value=0.1, format="%.2f")
        new_band = st.selectbox("Band", ["A", "B", "C", "D", "E"])
        new_energy = st.number_input("Previous Energy Reading", min_value=0.0)
        if st.form_submit_button("Add Feeder"):
            new_row = {
                "33/11KV INJECTION SUBSTATION": selected_sub,
                "FEEDER NAME": new_name,
                "PEAK LOAD EVER": new_peak,
                "BAND": new_band,
                "ENERGY READING (PREVIOUS)": new_energy,
                "FEEDER ID": df["FEEDER ID"].max() + 1
            }
            # Add to CSV or database (simplified here)
            st.success("Feeder added! (In prod: write to DB)")
            log_event("INFO", f"Admin added feeder: {new_name}", st.session_state.username)

# -------------------------------
# DASHBOARD (All Feeders Overview)
# -------------------------------
def live_dashboard():
    st.subheader("📈 Real-Time Grid Health")
    df = load_data(DATA_PATH)
    try:
        submitted = pd.read_csv(SUBMISSION_FILE)
        latest = submitted.groupby("FEEDER NAME").last().reset_index()
        merged = df[["FEEDER NAME", "PEAK LOAD EVER"]].merge(
            latest[["FEEDER NAME", "ENERGY READING (PRESENT)", "Timestamp"]],
            on="FEEDER NAME", how="left"
        )
        st.dataframe(merged)
        
        # Summary
        st.metric("Feeders Updated", (merged["Timestamp"].notna()).sum(), 
                  delta=f"/{len(merged)}")
    except:
        st.info("No submissions yet.")

# -------------------------------
# DATA ENTRY WITH SMART ALERTS
# -------------------------------
def data_entry():
    df = load_data(DATA_PATH)
    subs = get_substations()
    selected_sub = st.selectbox("🔌 Select Injection Substation", options=subs)

    sub_df = df[df["33/11KV INJECTION SUBSTATION"] == selected_sub].copy()
    hour_cols = [c for c in sub_df.columns if ":" in c]
    display_cols = ["FEEDER NAME", "PEAK LOAD EVER", "ENERGY READING (PREVIOUS)"] + hour_cols + ["ENERGY READING (PRESENT)"]
    editable = sub_df[display_cols].reset_index(drop=True)

    disabled_cols = ["FEEDER NAME", "PEAK LOAD EVER", "ENERGY READING (PREVIOUS)"]

    # Show alerts ABOVE table
    alerts = []
    for idx, row in editable.iterrows():
        feeder = row["FEEDER NAME"]
        peak = row["PEAK LOAD EVER"]
        for h in hour_cols:
            val = row[h]
            if pd.notna(val) and val > 0:
                if val > peak * 1.07:
                    alerts.append(f"🚨 {feeder} at {h}: Overload ({val:.2f} > {peak*1.07:.2f})")
        prev = row["ENERGY READING (PREVIOUS)"]
        pres = row.get("ENERGY READING (PRESENT)", np.nan)
        if pd.notna(pres) and pd.notna(prev):
            if pres > prev * 1.07:
                alerts.append(f"⚠️ Energy jump on {feeder}: {pres:.2f} > {prev*1.07:.2f}")

    if alerts:
        st.markdown("### ⚠️ System Alerts")
        for a in alerts[:10]:
            st.error(a)
        if AUDIO_ALERT and not st.session_state.get("alert_played", False):
            play_alert()
            st.session_state.alert_played = True
    else:
        st.session_state.alert_played = False

    # Editable table
    edited = st.data_editor(
        editable,
        disabled=disabled_cols,
        key="data_entry_table",
        height=600,
        use_container_width=True
    )

    if st.button("✅ Submit Data"):
        if alerts:
            st.warning("⚠️ Fix alerts before submission.")
        else:
            edited["Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            edited["Operator"] = st.session_state.username
            edited["Substation"] = selected_sub

            try:
                existing = pd.read_csv(SUBMISSION_FILE)
                combined = pd.concat([existing, edited], ignore_index=True)
            except FileNotFoundError:
                combined = edited
            combined.to_csv(SUBMISSION_FILE, index=False)
            log_event("SUCCESS", "Data submitted", st.session_state.username)
            st.success("✅ Data submitted to central grid log.")

# -------------------------------
# MAIN APP
# -------------------------------
st.set_page_config(page_title="⚡ Grid Nervous System", layout="wide")
st.title("⚡ Injection Substation Nervous System v2.0")

# Inject audio
inject_audio()

# Login gate
if not st.session_state.logged_in:
    login()
else:
    # Valid session
    page = render_sidebar()
    check_inactivity()  # every rerun

    if page == "📊 Data Entry":
        data_entry()
    elif page == "📈 Live Dashboard":
        live_dashboard()
    elif page == "⚙️ Feeder Manager":
        if st.session_state.role == "Admin":
            manage_feeders()
        else:
            st.error("🔒 Admin access required.")# app.py
# Smart Streamlit App for Hourly Load & Energy Entry (Operator Co-Pilot)
# Author: Yaquba + ChatGPT (GPT-5 Thinking) — 17+ yrs Python/Streamlit
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import io
import os
from typing import List
import base64
import streamlit.components.v1 as components

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="Smart Load & Energy Assistant", layout="wide", initial_sidebar_state="expanded")

DEFAULT_DATA_PATH = r"HOURLY LOAD SHEET.xlsx"  # You can change this or upload in Admin Tools
MASTER_CSV = "data_master.csv"                 # normalized copy of the master sheet
SUBMISSIONS_CSV = "submitted_data.csv"         # all submissions appended here
OPERATORS_CSV = "operators.csv"                # optional operator registry

IDLE_SECONDS_LIMIT = 17  # alarm threshold


# =============================================================================
# HELPERS
# =============================================================================
def get_hour_cols() -> List[str]:
    # Accept either "01:00:00"..."24:00:00" or "01:00"..."24:00"
    return [f"{h:02d}:00:00" for h in range(1, 25)]

def coalesce_hour_cols(df: pd.DataFrame) -> List[str]:
    hrs = []
    for c in df.columns:
        # pick columns that look like time (contain ':') and numeric-valued
        if ":" in str(c):
            hrs.append(c)
    # keep stable order by sorting by hour position if possible
    def parse_hr(x):
        s = str(x)
        # take first two digits before ':'
        try:
            return int(s.split(":")[0])
        except Exception:
            return 999
    hrs_sorted = sorted(hrs, key=parse_hr)
    return hrs_sorted

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize headers (trim)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Expected columns
    needed = {
        "33/11KV INJECTION SUBSTATION": "33/11KV INJECTION SUBSTATION",
        "FEEDER NAME": "FEEDER NAME",
        "PEAK LOAD EVER": "PEAK LOAD EVER",
        "ENERGY READING (PREVIOUS)": "ENERGY READING (PREVIOUS)",
        "ENERGY READING (PRESENT)": "ENERGY READING (PRESENT)",
    }
    # Attempt to auto-map close variants
    lower_map = {c.lower(): c for c in df.columns}
    for want in list(needed.keys()):
        if want not in df.columns:
            lw = want.lower()
            if lw in lower_map:
                needed[want] = lower_map[lw]
            else:
                # try looser heuristics
                candidates = [c for c in df.columns if lw.replace(" ", "") in c.lower().replace(" ", "")]
                if candidates:
                    needed[want] = candidates[0]
                else:
                    # create empty column if missing
                    df[want] = np.nan
                    needed[want] = want

    # Rename to canonical names
    renamer = {v: k for k, v in needed.items() if v != k}
    if renamer:
        df = df.rename(columns=renamer)

    # Coerce numeric for key numeric cols
    for c in ["PEAK LOAD EVER", "ENERGY READING (PREVIOUS)", "ENERGY READING (PRESENT)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Hour columns numeric
    for c in coalesce_hour_cols(df):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Strip feeder/substation text
    for c in ["33/11KV INJECTION SUBSTATION", "FEEDER NAME"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df

@st.cache_data(show_spinner=False)
def load_master_from_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    df = ensure_columns(df)
    return df

@st.cache_data(show_spinner=False)
def load_master_from_csv() -> pd.DataFrame:
    if os.path.exists(MASTER_CSV):
        df = pd.read_csv(MASTER_CSV)
        df = ensure_columns(df)
        return df
    else:
        # If CSV not present, try Excel default
        if os.path.exists(DEFAULT_DATA_PATH):
            return load_master_from_excel(DEFAULT_DATA_PATH)
        else:
            # empty starter
            cols = ["33/11KV INJECTION SUBSTATION", "FEEDER NAME", "PEAK LOAD EVER",
                    "ENERGY READING (PREVIOUS)", "ENERGY READING (PRESENT)"] + [f"{h:02d}:00:00" for h in range(1, 25)]
            return pd.DataFrame(columns=cols)

def save_master_to_csv(df: pd.DataFrame):
    df.to_csv(MASTER_CSV, index=False)

def append_submissions(df_submit: pd.DataFrame):
    if os.path.exists(SUBMISSIONS_CSV):
        old = pd.read_csv(SUBMISSIONS_CSV)
        out = pd.concat([old, df_submit], ignore_index=True)
    else:
        out = df_submit
    out.to_csv(SUBMISSIONS_CSV, index=False)

def style_top_alert(msg: str, level: str = "error"):
    # Big alert above table
    if level == "error":
        st.error(msg)
    elif level == "warning":
        st.warning(msg)
    else:
        st.info(msg)

def play_alarm_if(flag: bool, message: str = "⚠️ Operator idle! Please continue data entry."):
    """
    Shows a toast + (optional) plays a short beep using WebAudio in JS.
    Autoplays without needing an audio file.
    """
    if not flag:
        return
    st.toast(message, icon="⏰")
    beep_js = """
    <script>
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.type = 'sine';
        o.frequency.value = 880;
        o.connect(g);
        g.connect(ctx.destination);
        g.gain.setValueAtTime(0.0001, ctx.currentTime);
        g.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.05);
        o.start();
        g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 1.0);
        o.stop(ctx.currentTime + 1.1);
      } catch(e) { /* ignore autoplay blocks */ }
    </script>
    """
    components.html(beep_js, height=0, width=0)

def update_idle_timer(touched: bool):
    # store last interaction timestamp
    if "last_action_ts" not in st.session_state:
        st.session_state.last_action_ts = time.time()
    if touched:
        st.session_state.last_action_ts = time.time()

def is_idle_over_limit() -> bool:
    if "last_action_ts" not in st.session_state:
        st.session_state.last_action_ts = time.time()
        return False
    return (time.time() - st.session_state.last_action_ts) >= IDLE_SECONDS_LIMIT

def data_hash(df: pd.DataFrame) -> str:
    try:
        return str(pd.util.hash_pandas_object(df.fillna("__NaN__"), index=True).sum())
    except Exception:
        return str(hash(df.shape))

def validate_rows(df_edit: pd.DataFrame, hour_cols: List[str]) -> List[str]:
    alerts = []
    for _, row in df_edit.iterrows():
        feeder = row.get("FEEDER NAME", "UNKNOWN")
        peak = row.get("PEAK LOAD EVER", np.nan)
        prev = row.get("ENERGY READING (PREVIOUS)", np.nan)
        pres = row.get("ENERGY READING (PRESENT)", np.nan)

        # Hourly checks
        if pd.notna(peak):
            for h in hour_cols:
                val = row.get(h, np.nan)
                if pd.notna(val):
                    if val > peak:
                        alerts.append(f"🚨 {feeder} at {h}: exceeds PEAK LOAD EVER ({val} > {peak}).")
                    if val > peak * 1.07:
                        alerts.append(f"⚠️ {feeder} at {h}: exceeds +7% tolerance ({val} > {peak*1.07:.2f}).")

        # Energy checks
        if pd.notna(prev) and pd.notna(pres):
            if pres < prev:
                alerts.append(f"🚨 {feeder}: Energy Present ({pres}) < Previous ({prev}).")
            if pres > prev * 1.07:
                alerts.append(f"⚠️ {feeder}: Energy Present exceeds +7% tolerance ({pres} > {prev*1.07:.2f}).")

    return alerts

def variance_column(df_edit: pd.DataFrame) -> pd.Series:
    pres = pd.to_numeric(df_edit.get("ENERGY READING (PRESENT)"), errors="coerce")
    prev = pd.to_numeric(df_edit.get("ENERGY READING (PREVIOUS)"), errors="coerce")
    return (pres - prev).round(3)


# =============================================================================
# SESSION DEFAULTS
# =============================================================================
if "role" not in st.session_state:
    st.session_state.role = "Operator"
if "operator_name" not in st.session_state:
    st.session_state.operator_name = ""
if "selected_sub" not in st.session_state:
    st.session_state.selected_sub = None
if "data_hash" not in st.session_state:
    st.session_state.data_hash = ""


# =============================================================================
# SIDEBAR: LOGIN + NAV
# =============================================================================
st.sidebar.title("⚡ Smart Assistant")
st.sidebar.caption("Reliable entry • Real-time alerts • Fewer errors")

st.sidebar.subheader("Login")
st.session_state.role = st.sidebar.radio("Role", ["Operator", "Admin"], index=0)
st.session_state.operator_name = st.sidebar.text_input("Your Name / ID", value=st.session_state.operator_name)

# Sidebar Nav based on role
if st.session_state.role == "Operator":
    nav_pages = ["Data Entry", "My Feeders", "Individual Feeder", "Reports"]
else:
    nav_pages = ["Data Entry", "All Feeders", "Individual Feeder", "Reports", "Admin Tools"]

page = st.sidebar.radio("Navigation", nav_pages, index=0)
st.sidebar.write("---")
st.sidebar.info("Tip: Stay active while entering data. Idle > 17s triggers an alarm.")


# =============================================================================
# LOAD MASTER DATA
# =============================================================================
# Priority: cached CSV (Admin Tools can refresh it) -> Excel default if present
master_df = load_master_from_csv()
hour_cols = coalesce_hour_cols(master_df)
if not hour_cols:
    # if hour columns not present, create them
    hour_cols = [f"{h:02d}:00:00" for h in range(1, 25)]
    for c in hour_cols:
        if c not in master_df.columns:
            master_df[c] = np.nan

subs_list = sorted(master_df["33/11KV INJECTION SUBSTATION"].dropna().unique().tolist())

# =============================================================================
# TOP BANNER
# =============================================================================
st.markdown(
    """
    <div style="padding: 10px 16px; border-radius: 14px; background: linear-gradient(90deg, #0ea5e9, #22c55e); color: white; font-weight: 600;">
      Smart Load & Energy Assistant — Live Validation • +7% Tolerance • Idle Alarm
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# PAGE: DATA ENTRY (Substation-scoped, Operator-focused)
# =============================================================================
if page == "Data Entry":
    st.header("📊 Hourly Data Entry")

    if subs_list:
        selected_sub = st.selectbox("Select your 33/11KV Injection Substation", subs_list, index=0, key="sub_select")
    else:
        selected_sub = st.selectbox("Select your 33/11KV Injection Substation", ["— No substations —"], disabled=True)

    st.session_state.selected_sub = selected_sub

    sub_df = master_df[master_df["33/11KV INJECTION SUBSTATION"] == selected_sub].copy()

    display_cols = [
        "33/11KV INJECTION SUBSTATION",
        "FEEDER NAME",
        "PEAK LOAD EVER",
        "ENERGY READING (PREVIOUS)",
    ] + hour_cols + ["ENERGY READING (PRESENT)"]

    # Ensure display columns exist
    for c in display_cols:
        if c not in sub_df.columns:
            sub_df[c] = np.nan

    sub_df = sub_df[display_cols].reset_index(drop=True)

    # Non-editable columns
    disabled_cols = [
        "33/11KV INJECTION SUBSTATION",
        "FEEDER NAME",
        "PEAK LOAD EVER",
        "ENERGY READING (PREVIOUS)",
    ]

    st.subheader(f"Feeder Data — {selected_sub}")
    with st.container(border=True):
        st.caption("Enter hourly loads and the present energy reading. Alerts appear above the table and as popups.")
        edited = st.data_editor(
            sub_df,
            disabled=disabled_cols,
            num_rows="dynamic",
            use_container_width=True,
            key="data_entry_table",
        )

    # Touch timer if edited changed
    new_hash = data_hash(edited)
    update_idle_timer(touched=(new_hash != st.session_state.data_hash))
    st.session_state.data_hash = new_hash

    # Validations
    alerts = validate_rows(edited, hour_cols)

    # Show top alerts (above table)
    if alerts:
        style_top_alert("Validation Alerts Found:", "error")
        for a in alerts:
            st.write(a)
        st.toast("⚠️ Validation alerts found. Please review.", icon="🚨")
    else:
        st.success("✅ All entries are valid.")

    # Auto alarm if idle > 17s
    idle_flag = is_idle_over_limit()
    play_alarm_if(idle_flag)

    # Submission
    colA, colB = st.columns([1, 3])
    with colA:
        submit = st.button("💾 Submit Data", type="primary", use_container_width=True)
    with colB:
        st.caption("Submission requires no active alerts. Saved with timestamp, user & substation.")

    if submit:
        if alerts:
            st.warning("⚠️ Fix all alerts before submission.")
        else:
            out = edited.copy()
            out["Timestamp"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            out["Operator"] = st.session_state.operator_name or "Unknown"
            out["Role"] = st.session_state.role
            append_submissions(out)
            st.success("✅ Data submitted and saved successfully.")
            st.balloons()


# =============================================================================
# PAGE: MY FEEDERS (Operator view only my substation feeders quickly)
# =============================================================================
if page == "My Feeders":
    st.header("👤 My Feeders")
    if st.session_state.selected_sub:
        sdf = master_df[master_df["33/11KV INJECTION SUBSTATION"] == st.session_state.selected_sub].copy()
        st.dataframe(sdf[["33/11KV INJECTION SUBSTATION", "FEEDER NAME", "PEAK LOAD EVER", "ENERGY READING (PREVIOUS)", "ENERGY READING (PRESENT)"]], use_container_width=True)
    else:
        st.info("Select your substation on the Data Entry page first.")


# =============================================================================
# PAGE: ALL FEEDERS (Admin wide view)
# =============================================================================
if page == "All Feeders":
    st.header("🌍 All Feeders — Master View")
    st.dataframe(master_df, use_container_width=True)

    with st.expander("Export Master Data"):
        csv = master_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Master CSV", data=csv, file_name="data_master_export.csv", mime="text/csv")


# =============================================================================
# PAGE: INDIVIDUAL FEEDER (Drill-down)
# =============================================================================
if page == "Individual Feeder":
    st.header("🔎 Feeder Drill-down")
    sub = st.selectbox("Substation", subs_list, index=0 if subs_list else None, key="ff_sub")
    feeders = master_df[master_df["33/11KV INJECTION SUBSTATION"] == sub]["FEEDER NAME"].dropna().unique().tolist()
    feeder = st.selectbox("Feeder", feeders, index=0 if feeders else None, key="ff_feeder")

    fdf = master_df[(master_df["33/11KV INJECTION SUBSTATION"] == sub) & (master_df["FEEDER NAME"] == feeder)].copy()
    if fdf.empty:
        st.info("No data for this feeder.")
    else:
        row = fdf.iloc[0].to_dict()
        st.write(f"**Peak Load Ever:** {row.get('PEAK LOAD EVER', '—')}")
        st.write(f"**Prev Energy:** {row.get('ENERGY READING (PREVIOUS)', '—')}")
        st.write(f"**Present Energy:** {row.get('ENERGY READING (PRESENT)', '—')}")

        # Simple line chart of hourly values
        hrs = hour_cols
        y = fdf[hrs].T
        y.columns = ["Load"]
        y.index.name = "Hour"
        st.line_chart(y)


# =============================================================================
# PAGE: REPORTS (Daily/Monthly rollups from submissions)
# =============================================================================
if page == "Reports":
    st.header("📈 Reports & Summaries")
    if os.path.exists(SUBMISSIONS_CSV):
        sub = pd.read_csv(SUBMISSIONS_CSV)
        sub["Timestamp"] = pd.to_datetime(sub["Timestamp"], errors="coerce")
        sub["Date"] = sub["Timestamp"].dt.date
        st.subheader("Recent Submissions")
        st.dataframe(sub.tail(50), use_container_width=True)

        # Daily feeder count
        daily = sub.groupby(["Date", "33/11KV INJECTION SUBSTATION"]).size().reset_index(name="Entries")
        st.bar_chart(daily.pivot(index="Date", columns="33/11KV INJECTION SUBSTATION", values="Entries").fillna(0))
    else:
        st.info("No submissions yet. Submit from the Data Entry page.")


# =============================================================================
# PAGE: ADMIN TOOLS
# =============================================================================
if page == "Admin Tools":
    if st.session_state.role != "Admin":
        st.error("Admin access only.")
    else:
        st.header("🛠️ Admin Tools")

        st.subheader("1) Load/Refresh Master Data")
        uploaded = st.file_uploader("Upload Excel (.xlsx) or CSV to replace master data", type=["xlsx", "csv"])
        colu1, colu2 = st.columns([1,1])
        with colu1:
            if st.button("Use Default Excel Path", use_container_width=True):
                if os.path.exists(DEFAULT_DATA_PATH):
                    df_new = load_master_from_excel(DEFAULT_DATA_PATH)
                    save_master_to_csv(df_new)
                    st.success("Master data refreshed from default Excel.")
                else:
                    st.error(f"Default Excel not found at: {DEFAULT_DATA_PATH}")
        with colu2:
            if uploaded is not None:
                try:
                    if uploaded.name.lower().endswith(".xlsx"):
                        df_new = pd.read_excel(uploaded, sheet_name=0)
                    else:
                        df_new = pd.read_csv(uploaded)
                    df_new = ensure_columns(df_new)
                    save_master_to_csv(df_new)
                    st.success("Master data updated from uploaded file.")
                except Exception as e:
                    st.error(f"Failed to load uploaded file: {e}")

        st.divider()
        st.subheader("2) Add / Update Feeder")
        with st.form("add_feeder"):
            col1, col2 = st.columns(2)
            with col1:
                substation_in = st.text_input("33/11KV INJECTION SUBSTATION")
                feeder_in = st.text_input("FEEDER NAME")
                peak_in = st.number_input("PEAK LOAD EVER", min_value=0.0, step=0.1, value=0.0)
            with col2:
                prev_in = st.number_input("ENERGY READING (PREVIOUS)", min_value=0.0, step=0.1, value=0.0)
                pres_in = st.number_input("ENERGY READING (PRESENT)", min_value=0.0, step=0.1, value=0.0)
            submitted = st.form_submit_button("Add / Update Feeder", use_container_width=True)
        if submitted:
            dfm = load_master_from_csv()
            mask = (dfm["33/11KV INJECTION SUBSTATION"].str.lower() == substation_in.strip().lower()) & \
                   (dfm["FEEDER NAME"].str.lower() == feeder_in.strip().lower())
            new_row = {
                "33/11KV INJECTION SUBSTATION": substation_in.strip(),
                "FEEDER NAME": feeder_in.strip(),
                "PEAK LOAD EVER": peak_in,
                "ENERGY READING (PREVIOUS)": prev_in,
                "ENERGY READING (PRESENT)": pres_in,
            }
            # ensure hour cols exist
            for hc in hour_cols:
                if hc not in new_row:
                    new_row[hc] = np.nan

            if mask.any():
                # update
                dfm.loc[mask, list(new_row.keys())] = list(new_row.values())
                st.success("Feeder updated.")
            else:
                # add
                dfm = pd.concat([dfm, pd.DataFrame([new_row])], ignore_index=True)
                st.success("Feeder added.")

            save_master_to_csv(dfm)
            st.cache_data.clear()

        st.divider()
        st.subheader("3) Operator Registry (optional)")
        with st.expander("Upload/Download Operators CSV"):
            if os.path.exists(OPERATORS_CSV):
                ops_df = pd.read_csv(OPERATORS_CSV)
            else:
                ops_df = pd.DataFrame(columns=["Operator", "Substation"])
            st.dataframe(ops_df, use_container_width=True)
            csv_ops = ops_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Operators CSV", data=csv_ops, file_name="operators.csv", mime="text/csv")
            ops_upload = st.file_uploader("Upload Operators CSV", type=["csv"], key="ops_upload")
            if ops_upload is not None:
                try:
                    new_ops = pd.read_csv(ops_upload)
                    new_ops.to_csv(OPERATORS_CSV, index=False)
                    st.success("Operators CSV updated.")
                except Exception as e:
                    st.error(f"Failed to load operators CSV: {e}")

        st.divider()
        st.subheader("4) Safety & Validation Settings")
        tol = st.slider("Tolerance (%) above peak and energy previous", min_value=0, max_value=20, value=7, step=1)
        st.caption("Currently enforced in the app logic as fixed +7%. (Adjust and extend logic to use this slider if desired.)")

# =============================================================================
# FOOTER
# =============================================================================
st.write("")
st.caption("© KEDCO — Smart Assistant • Designed for reliability and zero-wrong entries.")
