"""
APL Logistics | Supply Chain Intelligence Dashboard
Delivery Performance, Delay Risk and Logistics Efficiency Analysis

Run from the project root:
    streamlit run dashboard/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ======================================================================
# PAGE CONFIGURATION (must be the first Streamlit call)
# ======================================================================
st.set_page_config(
    page_title="APL Logistics | Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# CONSTANTS
# ======================================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "outputs" / "cleaned_data" / "delivery_analysis_cleaned.csv"

COLUMN_ALIASES = {
    "performance": ["Delivery Performance", "delivery_performance"],
    "gap": ["Delivery Delay Gap", "delivery_delay_gap"],
    "risk": ["Late_delivery_risk", "Late Delivery Risk", "late_delivery_risk"],
    "shipping": ["Shipping Mode", "shipping_mode"],
    "region": ["Order Region", "order_region"],
    "market": ["Market", "market"],
    "segment": ["Customer Segment", "customer_segment"],
    "country": ["Order Country", "order_country"],
    "date": ["order date (DateOrders)", "Order Date (DateOrders)", "Order Date",
             "order_date", "order date"],
    "actual": ["Days for shipping (real)", "Actual Shipping Days",
               "actual_shipping_days", "days_for_shipping_real"],
    "scheduled": ["Days for shipment (scheduled)", "Scheduled Shipping Days",
                  "scheduled_shipping_days", "days_for_shipment_scheduled"],
}

FIELD_LABELS = {
    "performance": "Delivery Performance", "gap": "Delivery Delay Gap",
    "risk": "Late Delivery Risk", "shipping": "Shipping Mode",
    "region": "Order Region", "market": "Market", "segment": "Customer Segment",
    "country": "Order Country", "date": "Order Date",
    "actual": "Actual Shipping Days", "scheduled": "Scheduled Shipping Days",
}

STATUS_MAP = {
    "delayed": "Delayed", "late": "Delayed",
    "on-time": "On-time", "on time": "On-time", "ontime": "On-time",
    "early": "Early",
}
STATUS_ORDER = ["Early", "On-time", "Delayed"]
STATUS_COLORS = {"Early": "#60A5FA", "On-time": "#2563EB", "Delayed": "#EF4444"}
DELAY_SCALE = ["#10B981", "#F59E0B", "#EF4444"]
VOLUME_SCALE = ["#BFDBFE", "#60A5FA", "#1D4ED8"]
GOOD_SCALE = ["#EF4444", "#F59E0B", "#10B981"]

SHIP, AVG, DEL_PCT = "Shipments", "Avg Delay (days)", "Delayed %"
OK_RATE, RISK_PCT = "On-Time Rate %", "High Risk %"

PAGES = [
    "📊  Executive Overview",
    "⚠️  Delivery Risk",
    "🚚  Shipping Modes",
    "🌍  Regional Intelligence",
    "👥  Customer Intelligence",
    "🗂️  Data Explorer",
]

FILTERS = [
    ("f_ship", "_ship", "Shipping Mode"),
    ("f_region", "_region", "Order Region"),
    ("f_market", "_market", "Market"),
    ("f_segment", "_segment", "Customer Segment"),
    ("f_country", "_country", "Order Country"),
]

# ======================================================================
# THEME / CSS
# ======================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: #F8FAFC; color: #1E293B; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }

[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E2E8F0; box-shadow: 2px 0 14px rgba(15,23,42,.04); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #1E293B; }
.brand { display:flex; align-items:center; gap:12px; padding: 4px 0 14px 0; }
.brand-logo { width:42px; height:42px; border-radius:12px; background: linear-gradient(135deg,#3B82F6,#1D4ED8);
  display:flex; align-items:center; justify-content:center; font-size:22px; box-shadow:0 6px 16px rgba(37,99,235,.28); }
.brand-name { font-weight:800; font-size:16px; letter-spacing:.8px; color:#0F172A; line-height:1.1; }
.brand-sub { font-size:10px; letter-spacing:1.6px; color:#64748B; font-weight:600; }
.side-label { font-size:10.5px; font-weight:700; letter-spacing:1.6px; color:#64748B; text-transform:uppercase; margin:14px 0 6px 0; }
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] div[role="radiogroup"] label { padding: 9px 12px; border-radius: 10px; border: 1px solid transparent; transition: all .15s; }
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background:#F1F5F9; border-color:#E2E8F0; }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background:#EFF6FF; border-color:#BFDBFE; }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color:#1D4ED8; font-weight:700; }
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display:none; }

.page-title { font-size: 30px; font-weight: 800; color:#0F172A; letter-spacing:-.4px; margin-bottom:2px; }
.page-sub { color:#64748B; font-size:13.5px; margin-bottom:14px; }
.chip-row { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:18px; }
.chip { background:#FFFFFF; border:1px solid #E2E8F0; color:#475569; padding:4px 11px; border-radius:20px; font-size:11.5px; font-weight:600; box-shadow:0 1px 2px rgba(15,23,42,.04); }
.chip b { color:#2563EB; }

.section-title { font-size:17px; font-weight:700; color:#0F172A; margin:22px 0 4px 0; display:flex; align-items:center; gap:10px; }
.section-title::before { content:""; width:4px; height:18px; border-radius:2px; background:linear-gradient(180deg,#3B82F6,#1D4ED8); }
.section-cap { color:#64748B; font-size:12.5px; margin:0 0 12px 14px; }

.kpi-card { position:relative; overflow:hidden; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px;
  padding:18px 20px; min-height:128px; box-shadow:0 1px 3px rgba(15,23,42,.06), 0 8px 20px rgba(15,23,42,.05); margin-bottom:14px; }
.kpi-card::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--accent); }
.kpi-card::after { content:""; position:absolute; right:-30px; top:-30px; width:110px; height:110px; border-radius:50%;
  background: radial-gradient(circle, var(--accent) 0%, transparent 70%); opacity:.10; }
.kpi-top { display:flex; justify-content:space-between; align-items:center; }
.kpi-label { color:#64748B; font-size:11.5px; font-weight:700; letter-spacing:.9px; text-transform:uppercase; }
.kpi-icon { font-size:18px; width:34px; height:34px; display:flex; align-items:center; justify-content:center;
  border-radius:10px; background:#F1F5F9; border:1px solid #E2E8F0; }
.kpi-value { color:#0F172A; font-size:30px; font-weight:800; margin-top:10px; letter-spacing:-.5px; }
.kpi-note { color:#94A3B8; font-size:11.5px; margin-top:4px; }

.insight { display:flex; gap:12px; align-items:flex-start; background:#FFFFFF; border:1px solid #E2E8F0;
  border-radius:12px; padding:13px 16px; margin-bottom:9px; color:#334155; font-size:13.5px; line-height:1.5;
  box-shadow:0 1px 2px rgba(15,23,42,.04); }
.insight .dot { margin-top:7px; min-width:8px; height:8px; border-radius:50%; background:#2563EB; box-shadow:0 0 0 3px #DBEAFE; }
.insight b { color:#0F172A; }
.callout { border-radius:14px; padding:16px 18px; border:1px solid #E2E8F0; background:#FFFFFF; margin-bottom:10px;
  box-shadow:0 1px 3px rgba(15,23,42,.06), 0 6px 16px rgba(15,23,42,.04); }
.callout.good { border-left:4px solid #10B981; } .callout.bad { border-left:4px solid #EF4444; }
.callout .c-h { color:#64748B; font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; }
.callout .c-v { color:#0F172A; font-size:21px; font-weight:800; margin-top:4px; }
.callout .c-n { color:#64748B; font-size:12px; margin-top:2px; }
.note { color:#64748B; font-size:12px; font-style:italic; margin:6px 0 10px 2px; }

[data-testid="stVerticalBlockBorderWrapper"] { background:#FFFFFF; border-color:#E2E8F0 !important; border-radius:16px;
  box-shadow:0 1px 3px rgba(15,23,42,.05), 0 6px 18px rgba(15,23,42,.04); }
div[data-testid="stMetric"] { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:12px 14px; }
div[data-testid="stMetricLabel"] { color:#64748B; }
div[data-testid="stMetricValue"] { color:#0F172A; }
[data-testid="stTabs"] button[role="tab"] { font-weight:600; color:#64748B; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#2563EB; }
.stButton > button, .stDownloadButton > button { border-radius:10px; border:1px solid #BFDBFE; background:#EFF6FF; color:#1D4ED8; font-weight:600; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:#2563EB; background:#2563EB; color:#FFFFFF; }
.footer { text-align:center; color:#94A3B8; font-size:11px; padding:30px 0 6px 0; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def _streamlit_version():
    parts = []
    for p in st.__version__.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def stretch():
    """Width kwargs compatible with both old and new Streamlit versions."""
    if _streamlit_version() >= (1, 50):
        return {"width": "stretch"}
    return {"use_container_width": True}


# ======================================================================
# GENERIC HELPERS
# ======================================================================
def find_column(data, names):
    normalized = {str(c).strip().lower(): c for c in data.columns}
    for name in names:
        key = name.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def has(cols, key):
    return cols.get(key) is not None


def fmt_int(x):
    return f"{int(x):,}"


def fmt_pct(x, digits=2):
    return "N/A" if x is None or pd.isna(x) else f"{x:.{digits}f}%"


def fmt_days(x):
    return "N/A" if x is None or pd.isna(x) else f"{x:.2f} days"


def section(title, caption=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-cap">{caption}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def kpi_card_html(icon, label, value, desc, accent="#2563EB"):
    return (
        f'<div class="kpi-card" style="--accent:{accent}">'
        f'<div class="kpi-top"><span class="kpi-label">{label}</span><span class="kpi-icon">{icon}</span></div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{desc}</div></div>'
    )


def kpi_row(items):
    columns = st.columns(len(items))
    for col, item in zip(columns, items):
        with col:
            st.markdown(kpi_card_html(*item), unsafe_allow_html=True)


def insight_list(items):
    for text in items:
        st.markdown(f'<div class="insight"><span class="dot"></span><div>{text}</div></div>',
                    unsafe_allow_html=True)


def callout(kind, heading, value, desc):
    st.markdown(
        f'<div class="callout {kind}"><div class="c-h">{heading}</div>'
        f'<div class="c-v">{value}</div><div class="c-n">{desc}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height=360):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#1E293B", size=12),
        margin=dict(l=10, r=10, t=52, b=10),
        title=dict(font=dict(size=15, color="#0F172A"), x=0.01),
        legend=dict(title_text=""),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#2563EB", font_size=12, font_color="#1E293B"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1", tickfont=dict(color="#475569"))
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False, tickfont=dict(color="#475569"))
    return fig


def show_chart(fig, key, height=360):
    st.plotly_chart(style_fig(fig, height), key=key, **stretch())


def hbar(summary, value, label, title, key, fmt=".2f", scale=None, x_title=None):
    d = summary.sort_values(value, ascending=True)
    fig = px.bar(d, x=value, y=label, orientation="h", color=value, text=value,
                 color_continuous_scale=scale or DELAY_SCALE, title=title)
    fig.update_traces(texttemplate="%{text:" + fmt + "}", textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, xaxis_title=x_title or value, yaxis_title="")
    show_chart(fig, key, height=max(320, 30 * len(d) + 110))


def vbar(summary, x, y, title, key, fmt=".2f", scale=None, y_title=None):
    d = summary.sort_values(y, ascending=False)
    fig = px.bar(d, x=x, y=y, color=y, text=y, color_continuous_scale=scale or DELAY_SCALE, title=title)
    fig.update_traces(texttemplate="%{text:" + fmt + "}", textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, yaxis_title=y_title or y, xaxis_title="")
    show_chart(fig, key)


# ======================================================================
# DATA LAYER
# ======================================================================
@st.cache_data(show_spinner="Loading delivery dataset…")
def load_raw(path_str, modified_time):
    return pd.read_csv(path_str, low_memory=False)


@st.cache_data(show_spinner="Preparing analytics dataset…")
def prepare(raw):
    """Detect columns, apply the project's delay rules and add internal analysis columns."""
    cols = {key: find_column(raw, names) for key, names in COLUMN_ALIASES.items()}
    if cols["gap"] is None:
        return None, cols, len(raw)

    data = raw.copy()
    data["_gap"] = pd.to_numeric(data[cols["gap"]], errors="coerce")
    data = data.dropna(subset=["_gap"]).copy()

    derived = pd.Series(
        np.select([data["_gap"] > 0, data["_gap"] < 0], ["Delayed", "Early"], "On-time"),
        index=data.index,
    )
    if cols["performance"]:
        mapped = data[cols["performance"]].astype(str).str.strip().str.lower().map(STATUS_MAP)
        data["_status"] = mapped.fillna(derived)
    else:
        data["_status"] = derived

    data["_delayed"] = (data["_status"] == "Delayed").astype(int)
    data["_early"] = (data["_status"] == "Early").astype(int)
    data["_ontime"] = (data["_status"] == "On-time").astype(int)
    data["_ok"] = data["_early"] + data["_ontime"]

    data["_risk"] = (pd.to_numeric(data[cols["risk"]], errors="coerce")
                     if cols["risk"] else np.nan)

    if cols["date"]:
        data["_date"] = pd.to_datetime(data[cols["date"]], errors="coerce")
    for key, internal in [("shipping", "_ship"), ("region", "_region"), ("market", "_market"),
                          ("segment", "_segment"), ("country", "_country")]:
        if cols[key]:
            data[internal] = data[cols[key]].fillna("Unknown").astype(str).str.strip()
    for key, internal in [("actual", "_actual"), ("scheduled", "_sched")]:
        if cols[key]:
            data[internal] = pd.to_numeric(data[cols[key]], errors="coerce")
    return data, cols, len(raw)


def compute_kpis(df):
    n = len(df)
    delayed, early, ontime = int(df["_delayed"].sum()), int(df["_early"].sum()), int(df["_ontime"].sum())
    risk = df["_risk"].dropna()
    return {
        "total": n, "delayed": delayed, "early": early, "ontime": ontime,
        "on_time_rate": (early + ontime) / n * 100 if n else np.nan,
        "delayed_pct": delayed / n * 100 if n else np.nan,
        "early_pct": early / n * 100 if n else np.nan,
        "ontime_pct": ontime / n * 100 if n else np.nan,
        "avg_gap": df["_gap"].mean(),
        "risk_ratio": risk.mean() * 100 if len(risk) else None,
        "markets": df["_market"].nunique() if "_market" in df else None,
        "regions": df["_region"].nunique() if "_region" in df else None,
    }


def group_summary(df, key, label):
    g = (df.groupby(key)
         .agg(**{SHIP: ("_status", "size"), AVG: ("_gap", "mean"),
                 "Delayed Shipments": ("_delayed", "sum"), "ok_n": ("_ok", "sum"),
                 "risk_n": ("_risk", "sum"), "risk_c": ("_risk", "count")})
         .reset_index())
    g[DEL_PCT] = g["Delayed Shipments"] / g[SHIP] * 100
    g[OK_RATE] = g["ok_n"] / g[SHIP] * 100
    g[RISK_PCT] = g["risk_n"] / g["risk_c"].replace(0, np.nan) * 100
    total_delayed = g["Delayed Shipments"].sum()
    g["Share of All Delayed %"] = g["Delayed Shipments"] / total_delayed * 100 if total_delayed else 0.0
    return g.rename(columns={key: label}).drop(columns=["ok_n", "risk_n", "risk_c"])


def status_counts(df):
    counts = df["_status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    out = pd.DataFrame({"Status": counts.index, SHIP: counts.values})
    out["Share %"] = out[SHIP] / max(out[SHIP].sum(), 1) * 100
    return out


def status_by(df, key, label):
    t = df.groupby([key, "_status"]).size().reset_index(name=SHIP)
    return t.rename(columns={key: label, "_status": "Status"})


def monthly_trend(df):
    d = df.dropna(subset=["_date"])
    if d.empty:
        return pd.DataFrame()
    m = (d.groupby(d["_date"].dt.to_period("M"))
         .agg(**{SHIP: ("_status", "size"), AVG: ("_gap", "mean"), "ok": ("_ok", "mean")})
         .reset_index())
    m["Month"] = m["_date"].dt.to_timestamp()
    m[OK_RATE] = m["ok"] * 100
    return m.drop(columns=["_date", "ok"])


def extremes(df, internal, label):
    if internal not in df.columns or df[internal].nunique() < 2:
        return None
    s = group_summary(df, internal, label)
    return s.loc[s[AVG].idxmin()], s.loc[s[AVG].idxmax()], s


def severity_band(gap):
    return np.select([gap < 0, gap == 0, gap <= 1, gap <= 2],
                     ["Early", "On-time", "Delayed 1 day", "Delayed 2 days"], "Delayed 3+ days")


SEVERITY_ORDER = ["Early", "On-time", "Delayed 1 day", "Delayed 2 days", "Delayed 3+ days"]
SEVERITY_COLORS = {"Early": "#60A5FA", "On-time": "#2563EB", "Delayed 1 day": "#F59E0B",
                   "Delayed 2 days": "#F97316", "Delayed 3+ days": "#EF4444"}


# ======================================================================
# SIDEBAR (navigation + filters)
# ======================================================================
def reset_filters(bounds):
    for key, _, _ in FILTERS:
        st.session_state[key] = []
    if bounds:
        st.session_state["f_date"] = bounds


def render_sidebar(data):
    st.sidebar.markdown(
        '<div class="brand"><div class="brand-logo">📦</div><div>'
        '<div class="brand-name">APL LOGISTICS</div>'
        '<div class="brand-sub">CONTROL TOWER</div></div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="side-label">Navigation</div>', unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")

    st.sidebar.markdown('<div class="side-label">Data filters</div>', unsafe_allow_html=True)
    bounds = None
    if "_date" in data.columns and data["_date"].notna().any():
        bounds = (data["_date"].min().date(), data["_date"].max().date())

    for key, _, _ in FILTERS:
        st.session_state.setdefault(key, [])
    if bounds:
        st.session_state.setdefault("f_date", bounds)

    st.sidebar.button("↺  Reset filters", on_click=reset_filters, args=(bounds,), **stretch_btn())

    mask = pd.Series(True, index=data.index)
    active = []
    for key, internal, label in FILTERS:
        if internal in data.columns:
            options = sorted(data[internal].unique().tolist())
            chosen = st.sidebar.multiselect(label, options, key=key,
                                            placeholder=f"All {label.lower()} values",
                                            help=f"Leave empty to include every {label.lower()}.")
            if chosen:
                mask &= data[internal].isin(chosen)
                active.append(f"{label}: <b>{len(chosen)}</b>")

    if bounds:
        picked = st.sidebar.date_input("Order date range", min_value=bounds[0], max_value=bounds[1],
                                       key="f_date", help="Filters shipments by order date.")
        if isinstance(picked, (tuple, list)) and len(picked) == 2 and tuple(picked) != bounds:
            start = pd.Timestamp(picked[0])
            end = pd.Timestamp(picked[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mask &= data["_date"].between(start, end)
            active.append(f"Dates: <b>{picked[0]} → {picked[1]}</b>")

    filtered = data[mask]
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Showing **{len(filtered):,}** of **{len(data):,}** shipments")
    st.sidebar.caption("APL Logistics (KWE Group) · Unified Mentor Internship Project")
    return page, filtered, active


def stretch_btn():
    return {"use_container_width": True} if _streamlit_version() < (1, 50) else {"width": "stretch"}


def page_header(title, subtitle, active, n_filtered, n_total):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>',
                unsafe_allow_html=True)
    chips = [f'<span class="chip">Records: <b>{n_filtered:,}</b> / {n_total:,}</span>']
    chips += [f'<span class="chip">{a}</span>' for a in active] or ['<span class="chip">No filters applied</span>']
    st.markdown(f'<div class="chip-row">{"".join(chips)}</div>', unsafe_allow_html=True)


# ======================================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ======================================================================
def build_exec_insights(df):
    k = compute_kpis(df)
    out = [
        f"<b>{k['delayed']:,}</b> of <b>{k['total']:,}</b> shipments (<b>{k['delayed_pct']:.2f}%</b>) were delayed, "
        f"while <b>{k['on_time_rate']:.2f}%</b> were early or on-time.",
        f"The average delivery delay gap is <b>{k['avg_gap']:.2f} days</b> "
        f"(actual minus scheduled shipping days; positive means later than scheduled).",
    ]
    for internal, label, noun in [("_ship", "Shipping Mode", "shipping mode"),
                                  ("_region", "Order Region", "order region"),
                                  ("_segment", "Customer Segment", "customer segment")]:
        res = extremes(df, internal, label)
        if res:
            best, worst, _ = res
            out.append(f"Highest average delay by {noun}: <b>{worst[label]}</b> ({worst[AVG]:.2f} days). "
                       f"Lowest: <b>{best[label]}</b> ({best[AVG]:.2f} days).")
    if k["risk_ratio"] is not None:
        out.append(f"<b>{k['risk_ratio']:.2f}%</b> of shipments carry the existing Late_delivery_risk flag "
                   f"in this selection.")
    trend = monthly_trend(df) if "_date" in df.columns else pd.DataFrame()
    if len(trend) >= 2:
        first, last = trend.iloc[0], trend.iloc[-1]
        out.append(f"On-time rate moved from <b>{first[OK_RATE]:.2f}%</b> "
                   f"({first['Month']:%b %Y}) to <b>{last[OK_RATE]:.2f}%</b> ({last['Month']:%b %Y}).")
    return out


def page_executive(df, cols):
    k = compute_kpis(df)
    kpi_row([
        ("📦", "Total Shipments", fmt_int(k["total"]), "Records in current selection", "#2563EB"),
        ("✅", "On-Time Delivery Rate", fmt_pct(k["on_time_rate"]), "(Early + On-time) ÷ Total", "#3B82F6"),
        ("⏱️", "Delayed Shipments", fmt_int(k["delayed"]), f"{k['delayed_pct']:.2f}% of all shipments", "#EF4444"),
        ("⚡", "Early Shipments", fmt_int(k["early"]), f"{k['early_pct']:.2f}% of all shipments", "#38BDF8"),
    ])
    kpi_row([
        ("📈", "Average Delay Gap", fmt_days(k["avg_gap"]), "Actual − scheduled shipping days", "#F59E0B"),
        ("🚨", "Late Delivery Risk Ratio",
         fmt_pct(k["risk_ratio"]), "Share flagged in Late_delivery_risk" if k["risk_ratio"] is not None
         else "Risk column not available", "#F97316"),
        ("🛒", "Total Markets", fmt_int(k["markets"]) if k["markets"] is not None else "N/A",
         "Distinct markets in selection", "#2563EB"),
        ("🌐", "Total Regions", fmt_int(k["regions"]) if k["regions"] is not None else "N/A",
         "Distinct order regions in selection", "#3B82F6"),
    ])

    section("Delivery Performance Mix", "How shipments split across early, on-time and delayed outcomes.")
    sc = status_counts(df)
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        fig = px.pie(sc, names="Status", values=SHIP, hole=0.68, color="Status",
                     color_discrete_map=STATUS_COLORS, category_orders={"Status": STATUS_ORDER},
                     title="Delivery Performance Distribution")
        fig.update_traces(textinfo="percent+label", sort=False,
                          marker=dict(line=dict(color="#FFFFFF", width=3)))
        fig.add_annotation(text=f"<b>{k['total']:,}</b><br>Shipments", showarrow=False,
                           font=dict(size=17, color="#0F172A"))
        fig.update_layout(showlegend=False)
        show_chart(fig, "exec_donut")
    with c2, st.container(border=True):
        fig = px.bar(sc, x=SHIP, y="Status", orientation="h", color="Status", text="Share %",
                     color_discrete_map=STATUS_COLORS, category_orders={"Status": STATUS_ORDER[::-1]},
                     title="Delivery Status Distribution")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, yaxis_title="")
        show_chart(fig, "exec_status_bar")

    section("Performance Trend", "Monthly shipment volume, on-time rate and average delay by order date.")
    trend = monthly_trend(df) if "_date" in df.columns else pd.DataFrame()
    if trend.empty:
        st.info("Order date data is not available, so the trend view is hidden.")
    else:
        t1, t2 = st.columns(2)
        with t1, st.container(border=True):
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_bar(x=trend["Month"], y=trend[SHIP], name="Shipments", marker_color="#BFDBFE", secondary_y=False)
            fig.add_scatter(x=trend["Month"], y=trend[OK_RATE], name="On-Time Rate %", mode="lines+markers",
                            line=dict(color="#2563EB", width=3), secondary_y=True)
            fig.update_layout(title="Shipment Volume & On-Time Rate")
            fig.update_yaxes(title_text="Shipments", secondary_y=False)
            fig.update_yaxes(title_text="On-Time Rate %", secondary_y=True, showgrid=False)
            show_chart(fig, "exec_trend_1")
        with t2, st.container(border=True):
            fig = px.area(trend, x="Month", y=AVG, title="Average Delay Gap Over Time",
                          color_discrete_sequence=["#F59E0B"])
            fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
            fig.update_layout(yaxis_title="Avg delay gap (days)", xaxis_title="")
            show_chart(fig, "exec_trend_2")

    section("Executive Insights", "Generated directly from the current filter selection.")
    insight_list(build_exec_insights(df))


# ======================================================================
# PAGE 2 — DELIVERY RISK
# ======================================================================
def page_risk(df, cols):
    k = compute_kpis(df)
    kpi_row([
        ("⏱️", "Delayed Share", fmt_pct(k["delayed_pct"]), "Gap > 0", "#EF4444"),
        ("✅", "On-Time Share", fmt_pct(k["ontime_pct"]), "Gap = 0", "#2563EB"),
        ("⚡", "Early Share", fmt_pct(k["early_pct"]), "Gap < 0", "#3B82F6"),
        ("🚨", "High-Risk Flag Share", fmt_pct(k["risk_ratio"]),
         "Existing Late_delivery_risk column" if k["risk_ratio"] is not None else "Risk column not available",
         "#F59E0B"),
    ])

    section("Risk Flag Analysis", "Late_delivery_risk is an existing dataset field, not an independently trained model.")
    c1, c2 = st.columns(2)
    if k["risk_ratio"] is not None:
        rc = (df["_risk"].dropna().map({0: "Low / No Risk", 1: "High Risk"}).fillna("Other")
              .value_counts().reset_index())
        rc.columns = ["Risk", SHIP]
        with c1, st.container(border=True):
            fig = px.bar(rc, x="Risk", y=SHIP, color="Risk", text=SHIP, title="Late Delivery Risk Distribution",
                         color_discrete_map={"Low / No Risk": "#10B981", "High Risk": "#EF4444", "Other": "#94A3B8"})
            fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, xaxis_title="")
            show_chart(fig, "risk_dist")
        ct = df.dropna(subset=["_risk"]).copy()
        ct["Risk"] = ct["_risk"].map({0: "Low / No Risk", 1: "High Risk"}).fillna("Other")
        ct = ct.groupby(["Risk", "_status"]).size().reset_index(name=SHIP).rename(columns={"_status": "Status"})
        with c2, st.container(border=True):
            fig = px.bar(ct, x="Risk", y=SHIP, color="Status", barmode="relative", barnorm="percent",
                         color_discrete_map=STATUS_COLORS, category_orders={"Status": STATUS_ORDER},
                         title="Observed Delivery Status within Each Risk Flag (%)")
            fig.update_layout(yaxis_title="% of shipments", xaxis_title="")
            show_chart(fig, "risk_vs_status")
        flagged = df[df["_risk"] == 1]
        if len(flagged):
            insight_list([
                f"Among shipments flagged high risk, <b>{flagged['_delayed'].mean() * 100:.2f}%</b> were observed "
                f"as delayed (gap &gt; 0)."
            ])
    else:
        st.info("The Late_delivery_risk column is not available in this dataset.")

    section("Delay Gap Distribution", "Dashed line marks zero (delivered exactly as scheduled).")
    with st.container(border=True):
        fig = px.histogram(df, x="_gap", color="_status", nbins=30, barmode="overlay", opacity=0.85,
                           color_discrete_map=STATUS_COLORS, category_orders={"_status": STATUS_ORDER},
                           title="Distribution of Delivery Delay Gap (Actual − Scheduled Days)",
                           labels={"_status": "Status"})
        fig.add_vline(x=0, line_dash="dash", line_color="#94A3B8")
        fig.update_layout(xaxis_title="Delivery Delay Gap (days)", yaxis_title="Shipments")
        show_chart(fig, "risk_hist", 380)

    section("Delay Severity", "Analytical severity bands based on the delay gap in days (not contractual thresholds).")
    sev = pd.Series(severity_band(df["_gap"].to_numpy())).value_counts().reindex(SEVERITY_ORDER, fill_value=0)
    sev = pd.DataFrame({"Band": sev.index, SHIP: sev.values})
    sev["Share %"] = sev[SHIP] / max(sev[SHIP].sum(), 1) * 100
    s1, s2 = st.columns([3, 2])
    with s1, st.container(border=True):
        fig = px.bar(sev, x="Band", y=SHIP, color="Band", text="Share %", color_discrete_map=SEVERITY_COLORS,
                     title="Shipments by Delay Severity Band")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, xaxis_title="")
        show_chart(fig, "risk_severity")
    with s2, st.container(border=True):
        delayed = df[df["_delayed"] == 1]
        st.markdown("**Delayed shipment profile**")
        if delayed.empty:
            st.write("No delayed shipments in this selection.")
        else:
            st.metric("Average delay (delayed only)", f"{delayed['_gap'].mean():.2f} days")
            st.metric("Maximum delay", f"{delayed['_gap'].max():.0f} days")
            st.metric("Delayed by 3+ days", f"{(delayed['_gap'] > 2).mean() * 100:.2f}% of delayed")

    section("Actual vs Scheduled Shipping Duration")
    if "_actual" in df.columns and "_sched" in df.columns:
        a, s = df["_actual"].mean(), df["_sched"].mean()
        kpi_row([
            ("🚚", "Avg Actual Days", fmt_days(a), "Real shipping duration", "#2563EB"),
            ("🗓️", "Avg Scheduled Days", fmt_days(s), "Planned shipping duration", "#3B82F6"),
            ("↔️", "Average Difference", fmt_days(a - s), "Actual − scheduled", "#F59E0B"),
        ])
        v1, v2 = st.columns(2)
        with v1, st.container(border=True):
            if "_ship" in df.columns:
                m = df.groupby("_ship").agg(Actual=("_actual", "mean"), Scheduled=("_sched", "mean")).reset_index()
                m = m.melt(id_vars="_ship", var_name="Type", value_name="Days")
                fig = px.bar(m, x="_ship", y="Days", color="Type", barmode="group", text="Days",
                             color_discrete_map={"Actual": "#2563EB", "Scheduled": "#94A3B8"},
                             title="Average Actual vs Scheduled Days by Shipping Mode",
                             labels={"_ship": "Shipping Mode"})
                fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
            else:
                m = pd.DataFrame({"Type": ["Actual", "Scheduled"], "Days": [a, s]})
                fig = px.bar(m, x="Type", y="Days", color="Type", title="Average Actual vs Scheduled Days")
            show_chart(fig, "risk_actual_sched")
        with v2, st.container(border=True):
            fig = go.Figure()
            fig.add_histogram(x=df["_actual"], name="Actual", marker_color="#2563EB", opacity=0.75)
            fig.add_histogram(x=df["_sched"], name="Scheduled", marker_color="#94A3B8", opacity=0.75)
            fig.update_layout(barmode="overlay", title="Actual vs Scheduled Duration Distribution",
                              xaxis_title="Shipping days", yaxis_title="Shipments")
            show_chart(fig, "risk_actual_hist")
    else:
        st.info("Actual and scheduled shipping day columns were not found, so this comparison is hidden.")


# ======================================================================
# PAGE 3 — SHIPPING MODES
# ======================================================================
def page_shipping(df, cols):
    if "_ship" not in df.columns:
        st.info("The Shipping Mode column is not available in this dataset.")
        return
    s = group_summary(df, "_ship", "Shipping Mode")
    best, worst = s.loc[s[AVG].idxmin()], s.loc[s[AVG].idxmax()]

    section("Mode Performance Snapshot", "Best and worst modes are based on average delay gap in the current selection.")
    c1, c2, c3 = st.columns(3)
    with c1:
        callout("good", "Lowest average delay", str(best["Shipping Mode"]),
                f"{best[AVG]:.2f} days · {best[OK_RATE]:.2f}% early/on-time")
    with c2:
        callout("bad", "Highest average delay", str(worst["Shipping Mode"]),
                f"{worst[AVG]:.2f} days · {worst[DEL_PCT]:.2f}% delayed")
    with c3:
        top = s.loc[s[SHIP].idxmax()]
        callout("good", "Highest shipment volume", str(top["Shipping Mode"]),
                f"{int(top[SHIP]):,} shipments")

    section("Volume & Delay Comparison")
    a, b = st.columns(2)
    with a, st.container(border=True):
        hbar(s, SHIP, "Shipping Mode", "Shipment Volume by Shipping Mode", "ship_vol", ",.0f", VOLUME_SCALE)
    with b, st.container(border=True):
        hbar(s, AVG, "Shipping Mode", "Average Delay by Shipping Mode", "ship_avg", ".2f", x_title="Avg delay gap (days)")

    section("Delivery Performance by Mode")
    c, d = st.columns(2)
    with c, st.container(border=True):
        sb = status_by(df, "_ship", "Shipping Mode")
        fig = px.bar(sb, y="Shipping Mode", x=SHIP, color="Status", orientation="h", barmode="relative",
                     barnorm="percent", color_discrete_map=STATUS_COLORS, category_orders={"Status": STATUS_ORDER},
                     title="Delivery Performance Mix by Shipping Mode (%)")
        fig.update_layout(xaxis_title="% of shipments", yaxis_title="")
        show_chart(fig, "ship_mix", max(320, 70 * len(s) + 120))
    with d, st.container(border=True):
        g = s.melt(id_vars="Shipping Mode", value_vars=[DEL_PCT, OK_RATE], var_name="Metric", value_name="Percent")
        fig = px.bar(g, x="Shipping Mode", y="Percent", color="Metric", barmode="group", text="Percent",
                     color_discrete_map={DEL_PCT: "#EF4444", OK_RATE: "#2563EB"},
                     title="Delayed % vs Analytical On-Time Rate by Mode")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis_title="% of shipments", xaxis_title="")
        show_chart(fig, "ship_pct", max(320, 70 * len(s) + 120))
    note("On-time rate here is an analytical delivery performance measure (Early + On-time ÷ Total). "
         "The dataset contains no contractual SLA threshold, so this is not actual SLA compliance.")

    section("Shipping Mode Efficiency Ranking", "Ranked by on-time rate, then lowest average delay.")
    r = s.sort_values([OK_RATE, AVG], ascending=[False, True]).reset_index(drop=True)
    r.insert(0, "Rank", r.index + 1)
    show_cols = ["Rank", "Shipping Mode", SHIP, AVG, DEL_PCT, OK_RATE, RISK_PCT]
    st.dataframe(
        r[show_cols].round(2), hide_index=True, **stretch(),
        column_config={
            OK_RATE: st.column_config.ProgressColumn(OK_RATE, format="%.1f%%", min_value=0, max_value=100),
            DEL_PCT: st.column_config.ProgressColumn(DEL_PCT, format="%.1f%%", min_value=0, max_value=100),
            SHIP: st.column_config.NumberColumn(SHIP, format="%d"),
        },
    )


# ======================================================================
# PAGE 4 — REGIONAL INTELLIGENCE
# ======================================================================
def page_regional(df, cols):
    if not any(c in df.columns for c in ("_region", "_market", "_country")):
        st.info("No region, market or country columns are available in this dataset.")
        return

    tabs = st.tabs(["🌍 Regions", "📊 Markets", "🏳️ Countries", "🔥 Heatmap & Map"])

    with tabs[0]:
        if "_region" not in df.columns:
            st.info("Order Region column is not available.")
        else:
            rs = group_summary(df, "_region", "Order Region")
            best, worst = rs.loc[rs[AVG].idxmin()], rs.loc[rs[AVG].idxmax()]
            c1, c2 = st.columns(2)
            with c1:
                callout("bad", "Highest-delay region", str(worst["Order Region"]), f"{worst[AVG]:.2f} days average delay")
            with c2:
                callout("good", "Best-performing region", str(best["Order Region"]), f"{best[AVG]:.2f} days average delay")
            a, b = st.columns(2)
            with a, st.container(border=True):
                hbar(rs, AVG, "Order Region", "Average Delay by Order Region", "reg_avg", ".2f", x_title="Avg delay gap (days)")
            with b, st.container(border=True):
                hbar(rs, SHIP, "Order Region", "Shipment Volume by Order Region", "reg_vol", ",.0f", VOLUME_SCALE)
            with st.container(border=True):
                path = ["_market", "_region"] if "_market" in df.columns else ["_region"]
                tm = df.groupby(path).agg(**{SHIP: ("_status", "size"), AVG: ("_gap", "mean")}).reset_index()
                tm = tm.rename(columns={"_market": "Market", "_region": "Order Region"})
                fig = px.treemap(tm, path=[p for p in ["Market", "Order Region"] if p in tm.columns],
                                 values=SHIP, color=AVG, color_continuous_scale=DELAY_SCALE,
                                 title="Shipment Volume (size) and Average Delay (color)")
                fig.update_layout(margin=dict(l=5, r=5, t=52, b=5))
                show_chart(fig, "reg_treemap", 460)

    with tabs[1]:
        if "_market" not in df.columns:
            st.info("Market column is not available.")
        else:
            ms = group_summary(df, "_market", "Market")
            a, b = st.columns(2)
            with a, st.container(border=True):
                hbar(ms, AVG, "Market", "Average Delay by Market", "mkt_avg", ".2f", x_title="Avg delay gap (days)")
            with b, st.container(border=True):
                hbar(ms, OK_RATE, "Market", "On-Time Rate by Market (Efficiency)", "mkt_ok", ".2f",
                     GOOD_SCALE, x_title="On-time rate %")
            with st.container(border=True):
                g = ms.melt(id_vars="Market", value_vars=[DEL_PCT, OK_RATE], var_name="Metric", value_name="Percent")
                fig = px.bar(g, x="Market", y="Percent", color="Metric", barmode="group", text="Percent",
                             color_discrete_map={DEL_PCT: "#EF4444", OK_RATE: "#2563EB"},
                             title="Market Efficiency Comparison")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
                fig.update_layout(yaxis_title="% of shipments", xaxis_title="")
                show_chart(fig, "mkt_group")

    with tabs[2]:
        if "_country" not in df.columns:
            st.info("Order Country column is not available.")
        else:
            cs = group_summary(df, "_country", "Order Country")
            f1, f2, f3 = st.columns(3)
            with f1:
                metric = st.selectbox("Rank countries by", [AVG, DEL_PCT, OK_RATE, SHIP], key="country_metric")
            with f2:
                top_n = st.slider("Countries to display", 5, 30, 15, key="country_topn")
            with f3:
                min_ship = st.slider("Minimum shipments per country", 1, 500, 20, key="country_min",
                                     help="Hides countries with too few shipments for a stable average.")
            eligible = cs[cs[SHIP] >= min_ship]
            if eligible.empty:
                st.warning("No countries meet the minimum shipment threshold. Lower the slider.")
            else:
                asc = metric == OK_RATE
                top = eligible.sort_values(metric, ascending=asc).head(top_n)
                fmt = ",.0f" if metric == SHIP else ".2f"
                scale = VOLUME_SCALE if metric == SHIP else (GOOD_SCALE if metric == OK_RATE else DELAY_SCALE)
                with st.container(border=True):
                    hbar(top, metric, "Order Country", f"Top {len(top)} Countries by {metric}",
                         "ctry_bar", fmt, scale)
                st.dataframe(
                    top[["Order Country", SHIP, AVG, DEL_PCT, OK_RATE, RISK_PCT]].round(2),
                    hide_index=True, **stretch())

    with tabs[3]:
        if "_region" in df.columns and "_ship" in df.columns:
            with st.container(border=True):
                pv = df.pivot_table(index="_region", columns="_ship", values="_gap", aggfunc="mean")
                fig = px.imshow(pv, aspect="auto", color_continuous_scale=DELAY_SCALE, text_auto=".2f",
                                title="Average Delay Gap: Region × Shipping Mode",
                                labels=dict(x="Shipping Mode", y="Order Region", color="Avg delay"))
                show_chart(fig, "reg_heat", max(380, 28 * len(pv) + 140))
        else:
            st.info("Region and Shipping Mode columns are both needed for the heatmap.")
        if "_country" in df.columns:
            with st.container(border=True):
                try:
                    cs = group_summary(df, "_country", "Order Country")
                    fig = px.choropleth(cs, locations="Order Country", locationmode="country names",
                                        color=AVG, hover_data=[SHIP, DEL_PCT, OK_RATE],
                                        color_continuous_scale=DELAY_SCALE,
                                        title="Average Delay Gap by Country")
                    fig.update_geos(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=False,
                                    landcolor="#E2E8F0", showcountries=True, countrycolor="#FFFFFF",
                                    projection_type="natural earth")
                    show_chart(fig, "ctry_map", 480)
                    note("Countries whose names are not recognised by the map are not plotted; "
                         "see the Countries tab for the full ranking.")
                except Exception as error:
                    st.info(f"The geographic map could not be rendered ({error}).")


# ======================================================================
# PAGE 5 — CUSTOMER INTELLIGENCE
# ======================================================================
def page_customer(df, cols):
    if "_segment" not in df.columns:
        st.info("The Customer Segment column is not available in this dataset.")
        return
    cs = group_summary(df, "_segment", "Customer Segment")
    worst = cs.loc[cs[DEL_PCT].idxmax()]

    section("Segment Snapshot")
    kpi_row([
        ("👥", "Customer Segments", fmt_int(len(cs)), "Distinct segments in selection", "#2563EB"),
        ("📦", "Largest Segment", str(cs.loc[cs[SHIP].idxmax(), "Customer Segment"]),
         f"{int(cs[SHIP].max()):,} shipments", "#3B82F6"),
        ("📈", "Highest Avg Delay", str(cs.loc[cs[AVG].idxmax(), "Customer Segment"]),
         f"{cs[AVG].max():.2f} days", "#F59E0B"),
        ("⏱️", "Highest Delayed %", str(worst["Customer Segment"]), f"{worst[DEL_PCT]:.2f}% delayed", "#EF4444"),
    ])

    section("Volume & Delay by Segment")
    a, b = st.columns(2)
    with a, st.container(border=True):
        fig = px.pie(cs, names="Customer Segment", values=SHIP, hole=0.6,
                     color_discrete_sequence=["#2563EB", "#60A5FA", "#6366F1", "#0EA5E9", "#94A3B8"],
                     title="Shipment Volume by Customer Segment")
        fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color="#FFFFFF", width=3)))
        fig.update_layout(showlegend=False)
        show_chart(fig, "seg_vol")
    with b, st.container(border=True):
        vbar(cs, "Customer Segment", AVG, "Average Delay by Customer Segment", "seg_avg",
             y_title="Avg delay gap (days)")

    section("Performance Distribution & Delay Exposure")
    c, d = st.columns(2)
    with c, st.container(border=True):
        sb = status_by(df, "_segment", "Customer Segment")
        fig = px.bar(sb, x="Customer Segment", y=SHIP, color="Status", barmode="relative", barnorm="percent",
                     color_discrete_map=STATUS_COLORS, category_orders={"Status": STATUS_ORDER},
                     title="Delivery Performance Mix by Segment (%)")
        fig.update_layout(yaxis_title="% of shipments", xaxis_title="")
        show_chart(fig, "seg_mix")
    with d, st.container(border=True):
        g = cs.melt(id_vars="Customer Segment", value_vars=[DEL_PCT, "Share of All Delayed %"],
                    var_name="Metric", value_name="Percent")
        fig = px.bar(g, x="Customer Segment", y="Percent", color="Metric", barmode="group", text="Percent",
                     color_discrete_map={DEL_PCT: "#EF4444", "Share of All Delayed %": "#F59E0B"},
                     title="Delayed Shipment Exposure by Segment")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis_title="%", xaxis_title="")
        show_chart(fig, "seg_exposure")
    note("Delayed % = delayed shipments within the segment. Share of All Delayed % = the segment's "
         "contribution to all delayed shipments in the selection.")

    section("Segment Summary Table")
    st.dataframe(cs.round(2), hide_index=True, **stretch())


# ======================================================================
# PAGE 6 — DATA EXPLORER
# ======================================================================
def display_view(df, cols):
    view = df[[c for c in df.columns if not c.startswith("_")]].copy()
    if not cols["performance"]:
        view["Delivery Status"] = df["_status"]
    return view


def page_explorer(df, cols, raw_rows, total_rows):
    tab_table, tab_quality = st.tabs(["🗂️ Interactive Table", "🧪 Data Quality Summary"])
    view = display_view(df, cols)

    with tab_table:
        section("Advanced Filters", "These refine the sidebar selection for the table and download below.")
        f1, f2 = st.columns(2)
        with f1:
            statuses = st.multiselect("Delivery status", STATUS_ORDER, key="x_status",
                                      placeholder="All statuses")
            shown_cols = st.multiselect("Columns to display", view.columns.tolist(), key="x_cols",
                                        placeholder="All columns")
        with f2:
            lo, hi = float(df["_gap"].min()), float(df["_gap"].max())
            if lo < hi:
                gap_range = st.slider("Delivery delay gap range (days)", lo, hi, (lo, hi), key="x_gap")
            else:
                gap_range = (lo, hi)
                st.caption(f"All records share a delay gap of {lo:.0f} days.")
            row_limit = st.selectbox("Rows to display", [100, 500, 1000, 5000, 10000], index=1, key="x_rows")

        mask = df["_gap"].between(*gap_range)
        if statuses:
            mask &= df["_status"].isin(statuses)
        explorer = view[mask.to_numpy()]
        if shown_cols:
            explorer = explorer[shown_cols]

        k1, k2, k3 = st.columns(3)
        k1.metric("Records matching", f"{len(explorer):,}")
        k2.metric("Columns", f"{explorer.shape[1]:,}")
        k3.metric("Rows displayed", f"{min(len(explorer), row_limit):,}")

        if explorer.empty:
            st.warning("No records match the explorer filters.")
        else:
            st.dataframe(explorer.head(row_limit), height=420, **stretch())
            st.download_button("⬇  Download filtered CSV", explorer.to_csv(index=False).encode("utf-8"),
                               file_name="APL_Logistics_Filtered_Analysis.csv", mime="text/csv")

    with tab_quality:
        section("Data Quality Summary", "Based on the full cleaned dataset and the current sidebar selection.")
        with st.spinner("Computing data quality metrics…"):
            duplicates = int(view.duplicated().sum())
            missing = view.isna().sum()
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Rows in CSV", f"{raw_rows:,}")
        q2.metric("Rows analysed", f"{total_rows:,}", help="Rows with a valid Delivery Delay Gap.")
        q3.metric("Rows in selection", f"{len(df):,}")
        q4.metric("Duplicate rows (selection)", f"{duplicates:,}")
        if raw_rows != total_rows:
            st.warning(f"{raw_rows - total_rows:,} rows were excluded because the delay gap was missing "
                       f"or non-numeric.")

        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Detected analytical columns**")
            mapping = pd.DataFrame({
                "Analytical field": [FIELD_LABELS[k] for k in COLUMN_ALIASES],
                "Dataset column": [cols[k] if cols[k] else "Not found" for k in COLUMN_ALIASES],
            })
            st.dataframe(mapping, hide_index=True, **stretch())
        with m2:
            st.markdown("**Missing values by column**")
            mv = pd.DataFrame({"Column": missing.index, "Missing": missing.values,
                               "Missing %": (missing.values / max(len(view), 1) * 100).round(2),
                               "Type": view.dtypes.astype(str).values})
            mv = mv.sort_values("Missing", ascending=False)
            if mv["Missing"].sum() == 0:
                st.success("No missing values in the current selection.")
            st.dataframe(mv, hide_index=True, height=300, **stretch())


# ======================================================================
# MAIN
# ======================================================================
def main():
    inject_css()

    if not DATA_PATH.exists():
        st.error(f"Cleaned dataset not found.\n\nExpected at:\n\n`{DATA_PATH}`\n\n"
                 "Run your cleaning notebook to create it, or move the CSV to this location.")
        st.stop()

    try:
        raw = load_raw(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except Exception as error:
        st.error(f"Unable to load the dataset: {error}")
        st.stop()

    data, cols, raw_rows = prepare(raw)
    if data is None:
        st.error("The required column **Delivery Delay Gap** is missing from the cleaned CSV.")
        st.write("Available columns:", raw.columns.tolist())
        st.stop()
    if data.empty:
        st.error("The dataset has no rows with a valid Delivery Delay Gap.")
        st.stop()

    page, filtered, active = render_sidebar(data)

    titles = {
        PAGES[0]: ("Executive Overview", "Network-wide delivery performance at a glance"),
        PAGES[1]: ("Delivery Risk Intelligence", "Delay exposure, severity and schedule adherence"),
        PAGES[2]: ("Shipping Mode Intelligence", "Efficiency, volume and reliability by shipping mode"),
        PAGES[3]: ("Regional Intelligence", "Region, market and country delivery performance"),
        PAGES[4]: ("Customer Intelligence", "Segment-level volume, delay and exposure"),
        PAGES[5]: ("Data Explorer", "Inspect, filter, validate and export shipment records"),
    }
    title, subtitle = titles[page]
    page_header(title, f"APL Logistics · {subtitle}", active, len(filtered), len(data))

    if filtered.empty:
        st.warning("No shipment records match the current filters. Adjust the sidebar filters or press "
                   "**Reset filters**.")
        st.stop()

    try:
        if page == PAGES[0]:
            page_executive(filtered, cols)
        elif page == PAGES[1]:
            page_risk(filtered, cols)
        elif page == PAGES[2]:
            page_shipping(filtered, cols)
        elif page == PAGES[3]:
            page_regional(filtered, cols)
        elif page == PAGES[4]:
            page_customer(filtered, cols)
        else:
            page_explorer(filtered, cols, raw_rows, len(data))
    except Exception as error:
        st.error(f"This page could not be rendered: {error}")
        with st.expander("Technical details"):
            st.exception(error)

    st.markdown(
        '<div class="footer">APL Logistics (KWE Group) · Delivery Performance, Delay Risk & Logistics '
        'Efficiency Analysis<br>Built with Python, Pandas, NumPy, Plotly and Streamlit · '
        'Unified Mentor Internship Project</div>',
        unsafe_allow_html=True,
    )


main()
