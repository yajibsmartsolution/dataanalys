"""Enhanced KEDCO Data Sheets: Intelligent Load Flow Query & Analysis
Author: YAJIB SMART SOLUTIONS + ChatGPT (GPT-5 Thinking) — Expert-Level Enhancements
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re
from typing import List
from difflib import get_close_matches
import altair as alt

# ----------------- Configuration -----------------
st.set_page_config(page_title="YAJIB Intelligent Feeder Analytics", layout="wide")

MONTH_ALIASES = {
    "JANUARY": ["JAN", "JANUAR", "JANUARY", "JANU"],
    "FEBRUARY": ["FEB", "FEBUARY", "FEBRUARY"],
    "MARCH": ["MAR", "MARCH"],
    "APRIL": ["APR", "APRIL"],
    "MAY": ["MAY"],
    "JUNE": ["JUN", "JUNE"],
    "JULY": ["JUL", "JULY"],
    "AUGUST": ["AUG", "AUGUST"],
    "SEPTEMBER": ["SEP", "SEPT", "SEPTEMBER"],
    "OCTOBER": ["OCT", "OCTOBER"],
    "NOVEMBER": ["NOV", "NOVEMBER"],
    "DECEMBER": ["DEC", "DECEMBER"]
}
MONTHS = list(MONTH_ALIASES.keys())
BAND_ORDER = ["A", "B", "C", "D", "E"]

# ----------------- Utility functions -----------------
@st.cache_data
def list_sheet_months(uploaded_file) -> List[str]:
    try:
        xls = pd.ExcelFile(uploaded_file)
        found = set()
        for s in xls.sheet_names:
            su = s.strip().upper()
            for canonical, aliases in MONTH_ALIASES.items():
                if su in aliases or su == canonical:
                    found.add(canonical)
        return sorted(list(found), key=lambda m: MONTHS.index(m))
    except Exception:
        return []

@st.cache_data
def load_and_merge(uploaded_file) -> pd.DataFrame:
    xls = pd.ExcelFile(uploaded_file)
    df_list = []
    for sheet in xls.sheet_names:
        sheet_up = sheet.strip().upper()
        matched = None
        for canonical, aliases in MONTH_ALIASES.items():
            if sheet_up == canonical or sheet_up in aliases:
                matched = canonical
                break
        if not matched:
            continue
        try:
            df = pd.read_excel(uploaded_file, sheet_name=sheet, header=0)
            if df.empty:
                continue
            if 'FEEDER NAME' not in df.columns:
                feeder_cols = [c for c in df.columns if 'FEEDER' in str(c).upper() and 'NAME' in str(c).upper()]
                if feeder_cols:
                    df.rename(columns={feeder_cols[0]: 'FEEDER NAME'}, inplace=True)
                else:
                    feeder_cols = [c for c in df.columns if 'FEEDER' == str(c).strip().upper()]
                    if feeder_cols:
                        df.rename(columns={feeder_cols[0]: 'FEEDER NAME'}, inplace=True)
                    else:
                        continue
            df['MONTH'] = matched
            df_list.append(df)
        except Exception:
            continue
    if not df_list:
        return pd.DataFrame()
    merged = pd.concat(df_list, ignore_index=True)
    return merged

# Robust time parsing: convert many forms to hours (float)
def parse_time_to_hours(val) -> float:
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if s == '' or s.lower() in ['nan', 'none', 'na']:
        return 0.0
    s = s.replace('hrs', '').replace('hr', '').strip()
    m = re.match(r"^(\d{1,3}):(\d{1,2})$", s)
    if m:
        h = int(m.group(1)); mm = int(m.group(2))
        return h + mm/60.0
    m = re.match(r"^(\d{1,3})\.(\d{1,2})$", s)
    if m:
        h = int(m.group(1)); part = int(m.group(2))
        if part <= 59:
            return h + part/60.0
        else:
            return float(h) + float(part)/100.0
    if s.replace('.', '', 1).isdigit():
        return float(s)
    nums = re.findall(r"\d+", s)
    if nums:
        return float(nums[0])
    return 0.0

@st.cache_data
def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    dfc.columns = [str(c).strip() for c in dfc.columns]

    day_regex = re.compile(r"^\s*(\d{1,2})(st|nd|rd|th)\s*$", re.IGNORECASE)
    day_cols = [c for c in dfc.columns if day_regex.match(str(c))]

    for i in range(1, 32):
        col = f"{i}th" if i>3 else ("1st" if i==1 else "2nd" if i==2 else "3rd")
        if col not in dfc.columns:
            short = str(i)
            if short not in dfc.columns:
                dfc[col] = np.nan

    # --- Enhanced Feeder Name Cleaning ---
    if 'FEEDER NAME' in dfc.columns:
        dfc['FEEDER NAME'] = (dfc['FEEDER NAME']
                              .astype(str)
                              .str.strip()
                              .str.upper()
                              .str.replace(r'\s+', ' ', regex=True)  # collapse spaces
                              .str.replace(r'[^\w\s]', '', regex=True)  # remove dots, hyphens
                              .str.strip())
        # Optional: map common typos
        typo_map = {
            '33KV IDH': '33KV IDH',
            '33KV AIRPORT': '33KV AIRPORT',
            'IDH': '33KV IDH',
            'AIRPORT': '33KV AIRPORT'
        }
        dfc['FEEDER NAME'] = dfc['FEEDER NAME'].map(typo_map).fillna(dfc['FEEDER NAME'])

    if 'BAND' in dfc.columns:
        dfc['BAND'] = dfc['BAND'].astype(str).str.strip().str.upper()
        dfc.loc[~dfc['BAND'].isin(BAND_ORDER), 'BAND'] = ''

    day_cols_found = [c for c in dfc.columns if day_regex.match(str(c))]
    for c in day_cols_found:
        dfc[c + ' (hrs)'] = dfc[c].apply(parse_time_to_hours)

    # --- SAFE Supplied Hours ---
    if 'Supplied Hours' not in dfc.columns:
        hrs_cols = [c for c in dfc.columns if c.endswith(' (hrs)')]
        if hrs_cols:
            dfc['Supplied Hours'] = dfc[hrs_cols].sum(axis=1).round(2)
        else:
            dfc['Supplied Hours'] = 0.0

    if dfc['Supplied Hours'].dtype == 'timedelta64[ns]':
        dfc['Supplied Hours'] = dfc['Supplied Hours'].dt.total_seconds() / 3600
    else:
        dfc['Supplied Hours'] = pd.to_numeric(dfc['Supplied Hours'], errors='coerce').fillna(0)
    dfc['Supplied Hours'] = dfc['Supplied Hours'].round(2)

    # --- SAFE Available Hours ---
    if 'Available Hours' not in dfc.columns:
        num_days = dfc.get('NUM_DAYS', 30)
        dfc['Available Hours'] = (24 * pd.to_numeric(num_days, errors='coerce').fillna(30)).round(2)
    else:
        if dfc['Available Hours'].dtype == 'timedelta64[ns]':
            dfc['Available Hours'] = dfc['Available Hours'].dt.total_seconds() / 3600
        else:
            dfc['Available Hours'] = pd.to_numeric(dfc['Available Hours'], errors='coerce').fillna(0)
        dfc['Available Hours'] = dfc['Available Hours'].round(2)

    # --- SAFE Percentage Compliance ---
    if 'Percentage Compliance' not in dfc.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            comp = (dfc['Supplied Hours'] / dfc['Available Hours']) * 100
            dfc['Percentage Compliance'] = np.where(
                dfc['Available Hours'] > 0,
                np.round(comp, 2),
                0.0
            )
    else:
        dfc['Percentage Compliance'] = pd.to_numeric(dfc['Percentage Compliance'], errors='coerce').fillna(0).round(2)

    if 'MONTH' not in dfc.columns:
        dfc['MONTH'] = ''

    if 'MONTH' in dfc.columns:
        dfc['MONTH'] = pd.Categorical(dfc['MONTH'], categories=MONTHS, ordered=True)

    return dfc

# ----------------- Query helpers -----------------
def find_matching_feeders(df: pd.DataFrame, query: str, limit=10) -> List[str]:
    if df.empty or 'FEEDER NAME' not in df.columns:
        return []
    names = df['FEEDER NAME'].dropna().astype(str).unique().tolist()
    q = query.strip().upper()
    if not q:
        return sorted(names)[:limit]
    # Exact or substring (case-insensitive)
    matches = [n for n in names if q in n]
    if matches:
        return sorted(matches)[:limit]
    # Fuzzy with low cutoff
    close = get_close_matches(q, names, n=limit, cutoff=0.3)
    return close

def build_transposed(df: pd.DataFrame, feeder: str, months: List[str]) -> pd.DataFrame:
    dff = df[df['FEEDER NAME'].astype(str) == feeder]
    if months:
        dff = dff[dff['MONTH'].isin(months)]
    if dff.empty:
        return pd.DataFrame()
    day_regex = re.compile(r"^\s*(\d{1,2})(st|nd|rd|th)\s*$", re.IGNORECASE)
    day_cols = [c for c in df.columns if day_regex.match(str(c))]
    day_hours = [c + ' (hrs)' for c in day_cols if c + ' (hrs)' in dff.columns]

    index = ['MONTH', 'BAND'] + day_cols + ['Supplied Hours', 'Available Hours', 'Percentage Compliance']
    result = {}
    for _, row in dff.sort_values(['MONTH', 'BAND'] if 'BAND' in dff.columns else ['MONTH']).iterrows():
        label = f"{row['MONTH']} ({row.get('BAND','')})"
        vals = [row['MONTH'], row.get('BAND','')]
        for dc in day_cols:
            v = row.get(dc, '')
            if pd.isna(v) or str(v).strip()=='' or str(v).lower()=='nan':
                vals.append('00:00')
            else:
                vals.append(str(v))
        vals.append(row.get('Supplied Hours',''))
        vals.append(row.get('Available Hours',''))
        vals.append(row.get('Percentage Compliance',''))
        result[label] = vals
    return pd.DataFrame(result, index=index)

# ----------------- Visualization helpers -----------------
def colored_bar_chart(df_filtered: pd.DataFrame):
    bar = df_filtered.groupby('MONTH', as_index=False, observed=True)['Supplied Hours'].sum()
    chart = alt.Chart(bar).mark_bar().encode(
        x=alt.X('MONTH:N', sort=MONTHS, title='Month'),
        y=alt.Y('Supplied Hours:Q', title='Supplied Hours'),
        color=alt.Color('MONTH:N', scale=alt.Scale(scheme='category10'), legend=None),
        tooltip=[alt.Tooltip('MONTH:N', title='Month'), alt.Tooltip('Supplied Hours:Q', title='Supplied Hours')]
    ).properties(title='Supplied Hours per Month')
    return chart

def band_pie_chart(df_filtered: pd.DataFrame):
    if 'BAND' not in df_filtered.columns:
        return None
    pie = df_filtered.groupby('BAND', as_index=False, observed=True)['Supplied Hours'].sum()
    chart = alt.Chart(pie).mark_arc().encode(
        theta=alt.Theta(field='Supplied Hours', type='quantitative'),
        color=alt.Color('BAND:N', scale=alt.Scale(scheme='category10')),
        tooltip=[alt.Tooltip('BAND:N', title='Band'), alt.Tooltip('Supplied Hours:Q', title='Supplied Hours')]
    ).properties(title='Supplied Hours by Band')
    return chart

# ----------------- Export helpers -----------------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_csv(buf, index=True)
    return buf.getvalue()

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    try:
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=True, sheet_name='Transposed')
        return buf.getvalue()
    except Exception:
        raise

def to_pdf_bytes(df: pd.DataFrame, title: str = 'Report') -> bytes:
    try:
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
    except Exception:
        raise RuntimeError('reportlab not installed')
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(letter))
    c.setFont('Helvetica-Bold', 14)
    c.drawString(0.5*inch, 7.5*inch, title)
    x = 0.3*inch; y = 7.0*inch
    max_cols = 8
    cols = df.columns.tolist()[:max_cols]
    row_h = 0.25*inch
    c.setFont('Helvetica', 8)
    for i, col in enumerate(['index'] + cols):
        c.drawString(x + i*1.5*inch, y, str(col))
    y -= row_h
    for idx, row in df.reset_index().head(25).iterrows():
        for i, val in enumerate([row.name] + [row[c] for c in cols]):
            c.drawString(x + i*1.5*inch, y, str(val))
        y -= row_h
        if y < 0.5*inch:
            c.showPage(); y = 7.5*inch
    c.save()
    return buf.getvalue()

# ----------------- Global Utility Functions -----------------
def safe_timedelta_to_float(series: pd.Series) -> pd.Series:
    def convert_val(x):
        if pd.isna(x):
            return 0.0
        if isinstance(x, pd.Timedelta):
            return x.total_seconds() / 3600
        try:
            return float(x)
        except (ValueError, TypeError):
            return 0.0
    return series.apply(convert_val)

def safe_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].astype(str).replace({
            'nan': '',
            '<NA>': '',
            'None': '',
            'NaT': '',
            '0 days 00:00:00': '',
            '00:00:00': '00:00'
        })
    return df

# ----------------- Main app UI -----------------
st.title('⚡ YAJIB SMART — Intelligent Load Flow Query & Analytics')
st.markdown('Upload your monthly Excel workbook (sheets named like JAN, FEB, MARCH or JANUARY, etc.).')

uploaded = st.file_uploader('Upload Excel workbook', type=['xlsx', 'xls'])

if not uploaded:
    st.info('Waiting for an Excel file...')
    st.stop()

with st.spinner('Loading and cleaning data...'):
    raw = load_and_merge(uploaded)
    if raw.empty:
        st.error('Loaded workbook had no readable monthly sheets with FEEDER NAME.')
        st.stop()
    cleaned = clean_and_normalize(raw)

# Auto-detect available months and feeders
available_months = list_sheet_months(uploaded)
if not available_months:
    st.error('No recognizable monthly sheets found. Ensure sheets are named JAN, FEB, MAR... or full month names.')
    st.stop()

# Auto-detect all feeders
all_feeders = sorted(cleaned['FEEDER NAME'].dropna().astype(str).unique().tolist())
if not all_feeders:
    st.warning('No feeders found in the uploaded file.')
    st.stop()

# ==================== DEBUG: Show detected feeders ====================
with st.expander("🔍 Debug: View Detected Feeders", expanded=False):
    st.write("**Unique Feeder Names in Data:**")
    st.write(all_feeders)
    st.write(f"**Total Unique Feeders:** {len(all_feeders)}")
    st.write("**Sample Data:**")
    st.dataframe(cleaned[['FEEDER NAME', 'MONTH'] + [c for c in cleaned.columns if 'Supplied Hours' in c or 'BAND' in c]].head(10))

# ==================== SIDEBAR FILTERS (Define Early!) ====================
st.sidebar.header('Query & Filters')

# ✅ Define ALL variables before any logic uses them
feeder_query = st.sidebar.text_input('Feeder search (partial or exact)')
months_sel = st.sidebar.multiselect('Months', options=available_months, default=available_months)
bands = st.sidebar.multiselect('Bands (optional)', options=BAND_ORDER, default=BAND_ORDER)
view_mode = st.sidebar.radio('View mode', ['Transposed (summary)', 'Raw table', 'Metrics & Charts'])
chart_style = st.sidebar.selectbox('Chart style', ['Colored Lines','Scatter (jitter)','Block Grid Scatter'])

# ✅ Critical: Define compare_mode BEFORE using it
compare_mode = st.sidebar.checkbox('Enable Multi-Feeder Comparison', help="Compare 2 or more feeders at once")

run = st.sidebar.button('Run Analysis')

if not run:
    st.info('Configure filters in the sidebar and click "Run Analysis"')
    st.stop()

# Filter candidates based on search
if feeder_query.strip():
    candidates = find_matching_feeders(cleaned, feeder_query.strip(), limit=50)
    if not candidates:
        st.warning(f"No feeders match '{feeder_query}'. Try a different search.")
        st.stop()
else:
    candidates = all_feeders

if len(candidates) < 1:
    st.warning('No feeders matched your query.')
    st.stop()

# ==================== Feeder Selection ====================
if compare_mode:
    selected_feeders = st.sidebar.multiselect(
        'Select Feeders to Compare',
        options=candidates,
        default=candidates[:min(3, len(candidates))]  # up to 3
    )
    if len(selected_feeders) < 2:
        st.warning("Please select at least 2 feeders for comparison.")
        st.stop()
else:
    selected_feeder = st.selectbox('Select feeder', options=candidates)

# ==================== VIEW MODE HANDLING ====================
if compare_mode:
    st.header(f"🔍 Multi-Feeder Comparison: {', '.join(selected_feeders)}")

    # Filter data
    df_filtered = cleaned[cleaned['FEEDER NAME'].isin(selected_feeders)]
    if months_sel:
        df_filtered = df_filtered[df_filtered['MONTH'].isin(months_sel)]
    if 'BAND' in df_filtered.columns and bands:
        df_filtered = df_filtered[df_filtered['BAND'].isin(bands)]

    if df_filtered.empty:
        st.warning("No data for selected feeders and filters.")
        st.stop()

    # Build transposed tables
    transposed_dfs = {}
    for feeder in selected_feeders:
        transposed_dfs[feeder] = build_transposed(cleaned, feeder, months_sel)

    # Display side-by-side
    cols = st.columns(len(selected_feeders))
    for col, feeder in zip(cols, selected_feeders):
        with col:
            st.subheader(f"📋 {feeder}")
            st.dataframe(safe_df_for_display(transposed_dfs[feeder]))

    # Compute metrics
    metric_data = []
    for feeder in selected_feeders:
        dff = df_filtered[df_filtered['FEEDER NAME'] == feeder]
        total = safe_timedelta_to_float(dff['Supplied Hours']).sum()
        avg = safe_timedelta_to_float(dff['Supplied Hours']).mean()
        comp = dff['Percentage Compliance'].mean()
        metric_data.append([feeder, total, avg, comp])

    metric_df = pd.DataFrame(metric_data, columns=['Feeder', 'Total Supplied (hrs)', 'Avg Daily (hrs)', 'Avg Compliance (%)'])

    st.subheader("📊 Comparative Metrics")
    st.dataframe(metric_df.style.format({
        'Total Supplied (hrs)': '{:.2f}',
        'Avg Daily (hrs)': '{:.2f}',
        'Avg Compliance (%)': '{:.2f}'
    }).background_gradient(cmap='Blues', subset=['Total Supplied (hrs)']))

    # Overlaid line chart
    day_regex = re.compile(r"^(\d{1,2})(st|nd|rd|th)$", re.IGNORECASE)
    day_cols = [c for c in cleaned.columns if day_regex.match(str(c))]
    day_hours = [c + ' (hrs)' for c in day_cols if c + ' (hrs)' in cleaned.columns]

    if day_hours:
        data = []
        for feeder in selected_feeders:
            dff = df_filtered[df_filtered['FEEDER NAME'] == feeder]
            for _, row in dff.iterrows():
                for dc in day_cols:
                    day_num = int(re.search(r"(\d{1,2})", dc)[0])
                    hours = parse_time_to_hours(row.get(dc, 0))
                    data.append({'Day': day_num, 'Hours': hours, 'Feeder': feeder, 'Month': row['MONTH']})
        df_chart = pd.DataFrame(data)
        chart = alt.Chart(df_chart).mark_line(point=True).encode(
            x=alt.X('Day:O', title='Day of Month'),
            y=alt.Y('mean(Hours):Q', title='Avg Supplied Hours'),
            color=alt.Color('Feeder:N', scale=alt.Scale(scheme='set1')),
            tooltip=['Feeder:N', 'Day:O', 'mean(Hours):Q']
        ).properties(title='Daily Supply Comparison (Overlaid)')
        st.altair_chart(chart, use_container_width=True)

    # Insights
    st.markdown("### 🧠 Insights")
    best = metric_df.loc[metric_df['Total Supplied (hrs)'].idxmax()]['Feeder']
    st.success(f"🏆 {best} has the highest total supplied hours.")

else:
    # === Single Feeder Mode ===
    df_filtered = cleaned[cleaned['FEEDER NAME'] == selected_feeder]
    if months_sel:
        df_filtered = df_filtered[df_filtered['MONTH'].isin(months_sel)]
    if 'BAND' in df_filtered.columns and bands:
        df_filtered = df_filtered[df_filtered['BAND'].isin(bands)]

    if df_filtered.empty:
        st.warning('No data after applying filters.')
        st.stop()

    transposed = build_transposed(cleaned, selected_feeder, months_sel)

    supplied_float = safe_timedelta_to_float(df_filtered['Supplied Hours'])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Months analysed', len(df_filtered['MONTH'].unique()))
    with col2:
        st.metric('Total Supplied Hours', f"{supplied_float.sum():.2f}")
    with col3:
        st.metric('Avg Daily Supplied (hrs)', f"{supplied_float.mean():.2f}")
    with col4:
        st.metric('Max Supplied in a Day (hrs)', f"{supplied_float.max():.2f}")

    if view_mode == 'Transposed (summary)':
        st.subheader(f'Transposed — {selected_feeder}')
        st.dataframe(safe_df_for_display(transposed))
    elif view_mode == 'Raw table':
        st.subheader('Raw filtered table')
        st.dataframe(df_filtered)
    else:
        st.subheader('Visualizations & Insights')
        st.altair_chart(colored_bar_chart(df_filtered), use_container_width=True)
        pie_chart = band_pie_chart(df_filtered)
        if pie_chart is not None:
            st.altair_chart(pie_chart, use_container_width=True)

        st.markdown('### Smart Insights')
        low_comp = df_filtered[df_filtered['Percentage Compliance'] < 50]
        if not low_comp.empty:
            st.warning(f"Found {len(low_comp)} rows with Percentage Compliance < 50%. Consider investigation.")
            st.dataframe(low_comp[['MONTH','BAND','Supplied Hours','Available Hours','Percentage Compliance']])
        else:
            st.success('No major compliance anomalies detected (>=50% threshold).')

# ----------------- Export area -----------------
st.markdown('---')
st.header('Export & Print')

export_df = (transposed.copy() if not compare_mode 
             else pd.concat([safe_df_for_display(transposed_dfs[f]) for f in selected_feeders], axis=1, keys=selected_feeders))
export_df = export_df.fillna('')

export_filename_base = "_vs_".join(selected_feeders) if compare_mode else selected_feeder

export_col1, export_col2, export_col3 = st.columns([1,1,1])

with export_col1:
    csv_bytes = to_csv_bytes(export_df)
    st.download_button(
        'Download CSV',
        data=csv_bytes,
        file_name=f'{export_filename_base}_transposed.csv',
        mime='text/csv'
    )

with export_col2:
    try:
        excel_bytes = to_excel_bytes(export_df)
        st.download_button(
            'Download Excel',
            data=excel_bytes,
            file_name=f'{export_filename_base}_transposed.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception:
        st.warning('Excel export requires openpyxl. pip install openpyxl')

with export_col3:
    try:
        pdf_bytes = to_pdf_bytes(export_df, title=f'{export_filename_base} — Transposed Report')
        st.download_button(
            'Download PDF',
            data=pdf_bytes,
            file_name=f'{export_filename_base}_report.pdf',
            mime='application/pdf'
        )
    except Exception:
        st.info('PDF export requires reportlab (optional).')

st.markdown('---')
st.header('Expert-level Recommendations (How YAJIB SMART SOLUTIONS Can Enhance and Update Your Overall Data Management)')
st.markdown("""
We treat this challenge as building a data platform, not just a one-off file parser. That means:

Standardization: Define a consistent Excel schema (e.g., one sheet per month, fixed column names) and automatically ingest data into a lightweight database (SQLite or Postgres).

Data Quality: Add continuous validation and schema checks during upload — catching missing feeder IDs, malformed timestamps, or suspicious outliers immediately.

Scalable Computation: Offload heavier analytics (forecasting, clustering, large aggregations) into background worker jobs, with results made available to the Streamlit UI via APIs.

Intelligent Analytics: Apply lightweight machine learning:

Anomaly detection (e.g., Isolation Forest) to flag suspicious feeders or months.

Forecasting (SARIMA) for short-term supply expectations.

Automated Alerts: When compliance falls below threshold (e.g., < X% for Y consecutive days), auto-send email, Slack, or WhatsApp alerts.

Practical Quick Wins Already Delivered in This App:

Improved cleaning and preprocessing.

Multi-month queries and comparisons.

Interactive charts with export to CSV/Excel.

Highlighting of anomalies directly in the dashboard.
""")

st.markdown("- Generate a version that writes cleaned data to a SQLite DB and exposes REST endpoints.")
st.markdown("- Add a small forecasting demo (Prophet) for a single feeder.")
st.markdown("- Add email/Slack alerting for anomaly thresholds.")

st.success('Analysis complete — adjust filters and re-run to explore other feeders or months.')