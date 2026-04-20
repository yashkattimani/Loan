#!/usr/bin/env python3
"""
Mutual Funds Investment Dashboard — mfapi.in (free, no key needed)
Historical backtesting (lumpsum / SIP) + Future projection simulator
Run:  streamlit run mutual_funds.py
"""
import warnings
warnings.filterwarnings("ignore")

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mutual Funds Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

TODAY = date.today()

C = dict(
    navy="#1A2E4A", blue="#1B5299", gold="#C8912A",
    green="#27AE60", red="#C0392B", orange="#E67E22",
    purple="#8E44AD", teal="#16A085", gray="#7F8C8D",
    bg="#F5F7FA",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main .block-container { padding-top:1rem; padding-bottom:0.5rem; }
  .metric-card {
    background:white; border-radius:10px; padding:12px 16px;
    border-left:4px solid; box-shadow:0 2px 8px rgba(0,0,0,0.07);
    margin-bottom:8px; height:100%;
  }
  .metric-label {
    font-size:10px; font-weight:700; color:#7F8C8D;
    letter-spacing:0.6px; text-transform:uppercase;
  }
  .metric-value { font-size:18px; font-weight:700; margin:5px 0 3px; }
  .metric-sub   { font-size:10px; color:#95A5A6; line-height:1.4; }
  .section-header {
    font-size:13px; font-weight:700; color:#1A2E4A;
    border-bottom:2px solid #C8912A; padding-bottom:4px;
    margin-bottom:12px; margin-top:4px;
  }
  .fund-badge {
    display:inline-block; padding:3px 9px; border-radius:12px;
    font-size:10px; font-weight:700; letter-spacing:0.4px; margin:2px;
  }
  .mode-box {
    border-radius:10px; padding:14px 18px; margin-bottom:12px;
    border:1.5px solid;
  }
  .info-strip {
    background:#EBF5FB; border-left:4px solid #1B5299;
    border-radius:6px; padding:10px 14px; font-size:11px;
    color:#1A2E4A; margin-bottom:10px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_inr(x: float) -> str:
    ax = abs(x)
    if ax >= 1e7: return f"₹{x/1e7:.2f} Cr"
    if ax >= 1e5: return f"₹{x/1e5:.2f} L"
    if ax >= 1e3: return f"₹{x/1e3:.1f}K"
    return f"₹{x:.0f}"

def fmt_inr_full(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"₹{sign}{abs(x):,.0f}"

def cagr_pct(start_val: float, end_val: float, years: float) -> float:
    if years <= 0 or start_val <= 0 or end_val <= 0:
        return 0.0
    return ((end_val / start_val) ** (1.0 / years) - 1) * 100

def metric_card(label: str, value: str, sub: str, color: str) -> str:
    return (
        f'<div class="metric-card" style="border-left-color:{color}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color}">{value}</div>'
        f'<div class="metric-sub">{sub}</div>'
        f'</div>'
    )

# ─────────────────────────────────────────────────────────────────────────────
#  KNOWN AMC LIST  (for clean grouping)
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_AMCS = [
    "Aditya Birla Sun Life", "Axis", "Bandhan", "Bank of India",
    "Baroda BNP Paribas", "Canara Robeco", "DSP", "Edelweiss",
    "Franklin Templeton", "HDFC", "HSBC", "ICICI Prudential",
    "IDBI", "IDFC", "IL&FS", "Invesco India", "ITI",
    "JM Financial", "Kotak Mahindra", "LIC", "Mahindra Manulife",
    "Mirae Asset", "Motilal Oswal", "Navi", "Nippon India",
    "NJ", "PGIM India", "PPFAS", "Quant", "Quantum",
    "SBI", "Shriram", "Sundaram", "Tata", "Taurus",
    "Trust", "Union", "UTI", "WhiteOak Capital", "Zerodha",
]

def extract_amc(name: str) -> str:
    for amc in sorted(KNOWN_AMCS, key=len, reverse=True):
        if name.lower().startswith(amc.lower()):
            return amc
    parts = name.split()
    return " ".join(parts[:min(2, len(parts))])

# ─────────────────────────────────────────────────────────────────────────────
#  API CALLS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fund_list() -> list:
    r = requests.get("https://api.mfapi.in/mf", timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_nav(scheme_code: int) -> tuple:
    r = requests.get(f"https://api.mfapi.in/mf/{scheme_code}", timeout=20)
    r.raise_for_status()
    data = r.json()
    meta = data.get("meta", {})
    raw  = data.get("data", [])
    if not raw:
        return meta, pd.DataFrame()
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["nav"]).sort_values("date").reset_index(drop=True)
    return meta, df

@st.cache_data(ttl=3600, show_spinner=False)
def build_fund_df(raw_json: str) -> pd.DataFrame:
    import json
    fund_list = json.loads(raw_json)
    df = pd.DataFrame(fund_list)
    df = df.rename(columns={"schemeCode": "code", "schemeName": "name"})
    # Keep only active funds (have a live ISIN) and deduplicate by scheme code
    if "isinGrowth" in df.columns:
        df = df[df["isinGrowth"].notna()]
    df = df.drop_duplicates(subset=["code"])
    df = df[["code", "name"]].dropna(subset=["name"])
    df["name"] = df["name"].astype(str)
    df["amc"] = df["name"].apply(extract_amc)
    return df

# ─────────────────────────────────────────────────────────────────────────────
#  COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def nav_on_or_after(navs: pd.DataFrame, target: date):
    ts = pd.Timestamp(target)
    fwd = navs[navs["date"] >= ts]
    if not fwd.empty:
        return fwd.iloc[0]
    bwd = navs[navs["date"] < ts]
    return bwd.iloc[-1] if not bwd.empty else None

def compute_lumpsum(navs: pd.DataFrame, start: date, amount: float) -> dict | None:
    row0 = nav_on_or_after(navs, start)
    if row0 is None:
        return None
    start_nav = row0["nav"]
    start_dt  = row0["date"]
    units     = amount / start_nav

    period = navs[navs["date"] >= start_dt].copy()
    period["portfolio_value"] = units * period["nav"]
    period["invested"]        = amount
    period["gain"]            = period["portfolio_value"] - amount
    period["gain_pct"]        = period["gain"] / amount * 100

    end_val = period["portfolio_value"].iloc[-1]
    end_dt  = period["date"].iloc[-1]
    years   = max((end_dt - start_dt).days / 365.25, 0.001)

    return dict(
        inv_type="Lumpsum", invested=amount, units=units,
        start_nav=start_nav, end_nav=navs["nav"].iloc[-1],
        end_value=end_val, abs_return=end_val - amount,
        abs_pct=(end_val - amount) / amount * 100,
        cagr=cagr_pct(amount, end_val, years),
        years=years, period_df=period,
        start_date=start_dt.date(), end_date=end_dt.date(),
    )

def compute_sip(navs: pd.DataFrame, start: date, monthly: float,
                end: date | None = None, stepup_pct: float = 0.0) -> dict | None:
    if end is None:
        end = navs["date"].iloc[-1].date()

    rows, total_units, total_invested = [], 0.0, 0.0
    sip_amt = monthly
    cur = date(start.year, start.month, 1)

    while cur <= end:
        row = nav_on_or_after(navs, cur)
        if row is not None:
            nav    = row["nav"]
            units  = sip_amt / nav
            total_units   += units
            total_invested += sip_amt
            rows.append(dict(date=row["date"], nav=nav,
                             units_bought=units, cum_units=total_units,
                             cum_invested=total_invested, sip_amount=sip_amt))
        if cur.month == 12:
            next_cur = date(cur.year + 1, 1, 1)
        else:
            next_cur = date(cur.year, cur.month + 1, 1)
        if stepup_pct > 0 and next_cur.month == 1 and next_cur != date(start.year, 1, 1):
            sip_amt *= (1 + stepup_pct / 100)
        cur = next_cur

    if not rows:
        return None

    sip_df = pd.DataFrame(rows)

    # Build daily portfolio value across full period
    full = navs[navs["date"] >= sip_df["date"].min()].copy().reset_index(drop=True)
    cum_u, cum_i = [], []
    for d in full["date"]:
        prior = sip_df[sip_df["date"] <= d]
        if prior.empty:
            cum_u.append(0.0); cum_i.append(0.0)
        else:
            cum_u.append(prior["cum_units"].iloc[-1])
            cum_i.append(prior["cum_invested"].iloc[-1])

    full["cum_units"]       = cum_u
    full["cum_invested"]    = cum_i
    full["portfolio_value"] = full["cum_units"] * full["nav"]
    full["gain"]            = full["portfolio_value"] - full["cum_invested"]

    end_val = total_units * navs["nav"].iloc[-1]
    abs_ret = end_val - total_invested
    years   = max((sip_df["date"].iloc[-1] - sip_df["date"].iloc[0]).days / 365.25, 0.001)
    # Approximate XIRR: treat average invested capital as invested for half duration
    approx_cagr = cagr_pct(total_invested / 2, end_val, years / 2) if years > 0.5 else 0.0

    return dict(
        inv_type="SIP", monthly=monthly, installments=len(sip_df),
        invested=total_invested, units=total_units,
        end_value=end_val, abs_return=abs_ret,
        abs_pct=abs_ret / total_invested * 100 if total_invested else 0,
        cagr_approx=approx_cagr, years=years,
        sip_df=sip_df, full_period=full,
        start_date=sip_df["date"].iloc[0].date(),
        end_date=sip_df["date"].iloc[-1].date(),
    )

def historical_returns(navs: pd.DataFrame) -> dict:
    if navs.empty:
        return {}
    latest_nav  = navs["nav"].iloc[-1]
    latest_date = navs["date"].iloc[-1]
    out = {}
    for yrs in [1, 2, 3, 5, 7, 10]:
        target = latest_date - pd.DateOffset(years=yrs)
        past   = navs[navs["date"] <= target]
        if not past.empty:
            pnav = past["nav"].iloc[-1]
            out[f"{yrs}yr"] = round(cagr_pct(pnav, latest_nav, yrs), 2)
    return out

def project_future(amount: float, inv_type: str, rate_pct: float,
                   years: int, stepup_pct: float = 0.0) -> pd.DataFrame:
    rate_m = rate_pct / 100 / 12
    rows   = []
    if inv_type == "Lumpsum":
        for m in range(years * 12 + 1):
            rows.append(dict(month=m, year=m/12,
                             invested=amount,
                             value=amount * (1 + rate_pct/100) ** (m/12)))
    else:
        val, invested, sip = 0.0, 0.0, amount
        rows.append(dict(month=0, year=0, invested=0, value=0))
        for m in range(1, years * 12 + 1):
            val      = (val + sip) * (1 + rate_m)
            invested += sip
            rows.append(dict(month=m, year=m/12, invested=invested, value=val))
            if m % 12 == 0 and stepup_pct > 0:
                sip *= (1 + stepup_pct / 100)

    df = pd.DataFrame(rows)
    df["gain"]     = df["value"] - df["invested"]
    df["gain_pct"] = df["gain"] / df["invested"].replace(0, np.nan) * 100
    return df

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1A2E4A 0%,#1B5299 100%);
     padding:16px 24px;border-radius:12px;margin-bottom:14px;
     border-bottom:3px solid #C8912A;">
  <h2 style="color:white;margin:0;font-size:20px;letter-spacing:0.5px;">
    📊 MUTUAL FUNDS INVESTMENT DASHBOARD
  </h2>
  <p style="color:#C8912A;margin:3px 0 0;font-size:11px;letter-spacing:0.3px;">
    Powered by mfapi.in &nbsp;|&nbsp; Real NAV Data · Historical Backtesting · Future Projection
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD FUND LIST
# ─────────────────────────────────────────────────────────────────────────────
import json

with st.spinner("Loading active fund universe from mfapi.in…"):
    try:
        raw_funds = fetch_fund_list()
        fund_df   = build_fund_df(json.dumps(raw_funds))
    except Exception as e:
        st.error(f"Could not reach mfapi.in: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR — FUND SELECTION + PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="background:#1A2E4A;padding:14px 16px;border-radius:8px;margin-bottom:10px;">
      <div style="color:#C8912A;font-size:10px;font-weight:700;letter-spacing:1px;">
        MUTUAL FUNDS
      </div>
      <div style="color:white;font-size:13px;font-weight:600;margin-top:4px;">
        Investment Dashboard
      </div>
      <div style="color:#BDC3C7;font-size:11px;">
        {len(fund_df):,} funds · mfapi.in
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔍 Find a Fund")
    search_mode = st.radio("Search by", ["AMC → Fund Name", "Direct Fund Search"],
                           horizontal=True)

    selected_code, selected_name = None, None

    if search_mode == "AMC → Fund Name":
        amc_list = ["— Select AMC —"] + sorted(fund_df["amc"].unique())
        chosen_amc = st.selectbox("Fund House / AMC", amc_list)
        if chosen_amc != "— Select AMC —":
            sub = fund_df[fund_df["amc"] == chosen_amc].reset_index(drop=True)
            names = sub["name"].tolist()
            codes = sub["code"].tolist()
            fidx  = st.selectbox("Fund Name", range(len(names)),
                                  format_func=lambda i: names[i])
            selected_code = int(codes[fidx])
            selected_name = names[fidx]
    else:
        term = st.text_input("Search fund name", placeholder="e.g. HDFC Flexi Cap Growth")
        if term:
            hits = fund_df[fund_df["name"].str.contains(term, case=False, na=False)].reset_index(drop=True)
            if hits.empty:
                st.warning("No matching funds found.")
            else:
                names = hits["name"].tolist()
                codes = hits["code"].tolist()
                fidx  = st.selectbox("Select fund", range(len(names)),
                                      format_func=lambda i: names[i])
                selected_code = int(codes[fidx])
                selected_name = names[fidx]

    if selected_code:
        st.divider()

        # ── Mode ──────────────────────────────────────────────────────────────
        st.markdown("### 🗂️ Dashboard Mode")
        mode = st.radio(
            "Choose mode",
            ["📅 Historical Backtest", "🔮 Future Projection"],
            help="Historical: simulate using real past NAVs  |  Future: project with chosen return rate",
        )

        st.divider()

        # ── Investment parameters ──────────────────────────────────────────────
        st.markdown("### 💰 Investment")
        inv_type = st.radio("Type", ["Lumpsum", "SIP (Monthly)"], horizontal=True)

        if inv_type == "Lumpsum":
            amount = st.number_input(
                "Amount (₹)", min_value=1_000, max_value=100_000_000,
                value=1_00_000, step=1_000, format="%d",
            )
            stepup = 0.0
        else:
            amount = st.number_input(
                "Monthly SIP (₹)", min_value=500, max_value=10_000_000,
                value=5_000, step=500, format="%d",
            )
            stepup = st.slider("Annual SIP Step-up %", 0, 30, 0,
                               help="Increase SIP by this % each year")

        st.divider()

        # ── Mode-specific params ───────────────────────────────────────────────
        if "Historical" in mode:
            st.markdown("### 📅 Period")
            start_date = st.date_input(
                "Invest from",
                value=date(TODAY.year - 5, TODAY.month, 1),
                min_value=date(1995, 1, 1),
                max_value=TODAY - timedelta(days=30),
                help="Your investment start date (use a past date)",
            )
            if inv_type == "SIP (Monthly)":
                end_date = st.date_input(
                    "SIP end date (or today)",
                    value=TODAY,
                    min_value=start_date + timedelta(days=30),
                    max_value=TODAY,
                )
            else:
                end_date = TODAY
        else:
            st.markdown("### 🔮 Projection Settings")
            proj_years = st.slider("Investment horizon (years)", 1, 40, 10)
            # Return rate picker — will be populated after NAV fetch below

# ─────────────────────────────────────────────────────────────────────────────
#  LANDING SCREEN (no fund selected)
# ─────────────────────────────────────────────────────────────────────────────
if not selected_code:
    st.markdown('<div class="section-header">👈 Select a fund from the sidebar to begin</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-strip">
    <b>How to use this dashboard:</b><br>
    1. Search for a mutual fund by AMC name or directly by fund name.<br>
    2. Choose <b>Historical Backtest</b> to see how a past investment would have grown using real NAV data.<br>
    3. Choose <b>Future Projection</b> to simulate returns for a chosen horizon using CAGR estimates.<br>
    4. Toggle between <b>Lumpsum</b> and <b>SIP</b> investment modes.
    </div>
    """, unsafe_allow_html=True)

    pop = {
        "HDFC Flexi Cap Fund": 101762,
        "Axis Large Cap Fund": 112277,
        "Mirae Asset Large & Midcap Fund": 112932,
        "PPFAS Flexi Cap Fund": 122640,
        "SBI Small Cap Fund": 125494,
        "Nippon India Small Cap Fund": 113177,
        "quant Multi Cap Fund": 100631,
        "Kotak Small Cap Fund": 102875,
    }
    st.markdown("#### 🌟 Popular Funds (click code in sidebar → Direct Search)")
    cols = st.columns(4)
    for i, (fname, code) in enumerate(pop.items()):
        with cols[i % 4]:
            st.markdown(
                f'<div class="metric-card" style="border-left-color:{C["blue"]}">'
                f'<div class="metric-label">Code: {code}</div>'
                f'<div class="metric-value" style="font-size:12px;color:{C["blue"]}">{fname}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  FETCH NAV DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching NAV history for {selected_name[:50]}…"):
    try:
        meta, navs = fetch_nav(selected_code)
    except Exception as e:
        st.error(f"Failed to fetch NAV: {e}")
        st.stop()

if navs.empty:
    st.error("No NAV data available for this fund.")
    st.stop()

hist_ret  = historical_returns(navs)
latest_nav = navs["nav"].iloc[-1]
nav_start  = navs["date"].iloc[0].date()
nav_end    = navs["date"].iloc[-1].date()
nav_years  = (navs["date"].iloc[-1] - navs["date"].iloc[0]).days / 365.25

# ─────────────────────────────────────────────────────────────────────────────
#  FUND HEADER
# ─────────────────────────────────────────────────────────────────────────────
h1, h2, h3, h4 = st.columns([4, 1.2, 1.2, 1.2])
with h1:
    st.markdown(f"### {selected_name}")
    badges = []
    for key, bg, fg in [
        ("scheme_category", "#EBF5FB", "#1B5299"),
        ("scheme_type",     "#EAFAF1", "#27AE60"),
        ("fund_house",      "#FEF9E7", "#C8912A"),
    ]:
        val = meta.get(key, "")
        if val:
            badges.append(
                f'<span class="fund-badge" style="background:{bg};color:{fg};">{val}</span>'
            )
    st.markdown(" ".join(badges), unsafe_allow_html=True)
    st.caption(
        f"Scheme: **{selected_code}** &nbsp;·&nbsp; "
        f"Data: {nav_start.strftime('%d %b %Y')} → {nav_end.strftime('%d %b %Y')} "
        f"({nav_years:.1f} yrs, {len(navs):,} records)"
    )

with h2:
    nav_prev = navs["nav"].iloc[-2] if len(navs) > 1 else latest_nav
    nav_delta = latest_nav - nav_prev
    st.metric("Latest NAV", f"₹{latest_nav:.4f}",
              delta=f"{nav_delta:+.4f} ({nav_delta/nav_prev*100:+.2f}%)")

with h3:
    best_ret_label = max(hist_ret, key=lambda k: int(k[:-2])) if hist_ret else None
    if "1yr" in hist_ret:
        color_1yr = C["green"] if hist_ret["1yr"] >= 0 else C["red"]
        st.metric("1yr Return", f"{hist_ret['1yr']:.2f}%")

with h4:
    if "3yr" in hist_ret:
        st.metric("3yr CAGR", f"{hist_ret['3yr']:.2f}%")

# ── Historical return strip ────────────────────────────────────────────────────
if hist_ret:
    st.markdown('<div class="section-header">Historical CAGR Returns</div>',
                unsafe_allow_html=True)
    ret_cols = st.columns(len(hist_ret))
    for col, (period, ret) in zip(ret_cols, hist_ret.items()):
        color = C["green"] if ret >= 0 else C["red"]
        with col:
            st.markdown(
                f'<div class="metric-card" style="border-left-color:{color};text-align:center;">'
                f'<div class="metric-label">{period} CAGR</div>'
                f'<div class="metric-value" style="color:{color}">{ret:.2f}%</div>'
                f'<div class="metric-sub">annualised</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  ██████  HISTORICAL BACKTEST MODE
# ─────────────────────────────────────────────────────────────────────────────
if "Historical" in mode:
    st.markdown('<div class="section-header">📅 Historical Backtest — Real NAV Data</div>',
                unsafe_allow_html=True)

    # ── Compute ───────────────────────────────────────────────────────────────
    if inv_type == "Lumpsum":
        result = compute_lumpsum(navs, start_date, amount)
    else:
        result = compute_sip(navs, start_date, amount, end_date, stepup)

    if result is None:
        st.error("No NAV data available from the selected start date. Try a later date.")
        st.stop()

    invested  = result["invested"]
    end_val   = result["end_value"]
    abs_ret   = result["abs_return"]
    abs_pct   = result["abs_pct"]
    years_    = result["years"]
    s_date    = result["start_date"]
    e_date    = result.get("end_date", nav_end)

    if inv_type == "Lumpsum":
        cagr_val = result["cagr"]
        subtitle = f"₹{amount:,} one-time · {s_date} → {e_date}"
    else:
        cagr_val = result["cagr_approx"]
        subtitle = (f"₹{amount:,}/mo × {result['installments']} instalments"
                    f" · {s_date} → {e_date}")

    gain_color = C["green"] if abs_ret >= 0 else C["red"]

    # ── KPI row ───────────────────────────────────────────────────────────────
    kpi = st.columns(5)
    kpis = [
        ("Invested",       fmt_inr(invested),    subtitle[:38] + "…" if len(subtitle) > 38 else subtitle,  C["blue"]),
        ("Current Value",  fmt_inr(end_val),     f"As of {e_date.strftime('%d %b %Y')}",                   C["teal"]),
        ("Absolute Return",fmt_inr(abs(abs_ret)), f"{abs_pct:+.2f}%",                                      gain_color),
        ("CAGR (Est.)",    f"{cagr_val:.2f}%",   "Compound annual growth rate",                             C["purple"]),
        ("Duration",       f"{years_:.1f} yrs",  f"{int(years_*12)} months",                               C["gold"]),
    ]
    for col, (lbl, val, sub, clr) in zip(kpi, kpis):
        with col:
            st.markdown(metric_card(lbl, val, sub, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main NAV + Portfolio Value Chart ──────────────────────────────────────
    if inv_type == "Lumpsum":
        pf   = result["period_df"]
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])

        fig1.add_trace(go.Scatter(
            x=navs["date"], y=navs["nav"],
            name="NAV (full history)", line=dict(color=C["gray"], width=1),
            opacity=0.35, showlegend=True,
        ), secondary_y=False)

        fig1.add_trace(go.Scatter(
            x=pf["date"], y=pf["nav"],
            name="NAV (investment period)", line=dict(color=C["blue"], width=2),
            showlegend=True,
        ), secondary_y=False)

        fig1.add_trace(go.Scatter(
            x=pf["date"], y=pf["portfolio_value"],
            name="Portfolio Value", line=dict(color=C["green"], width=2.5),
            fill="tozeroy", fillcolor="rgba(39,174,96,0.08)",
        ), secondary_y=True)

        fig1.add_trace(go.Scatter(
            x=pf["date"], y=pf["invested"],
            name="Invested", line=dict(color=C["orange"], width=1.5, dash="dash"),
        ), secondary_y=True)

        # Mark start point
        fig1.add_vline(x=str(s_date), line_dash="dot", line_color=C["gold"],
                       annotation_text=f"Buy {fmt_inr(amount)}", annotation_font_size=10)

        fig1.update_layout(
            title=dict(text=f"NAV & Portfolio Growth · {selected_name[:55]}", font_size=13),
            height=380, hovermode="x unified",
            legend=dict(orientation="h", y=1.08, x=0, font_size=10),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=60, b=40, l=50, r=50),
        )
        fig1.update_xaxes(showgrid=False)
        fig1.update_yaxes(title_text="NAV (₹)", secondary_y=False, showgrid=True,
                          gridcolor="#ECF0F1")
        fig1.update_yaxes(title_text="Portfolio Value (₹)", secondary_y=True)
        st.plotly_chart(fig1, use_container_width=True)

    else:  # SIP
        fp   = result["full_period"]
        sipd = result["sip_df"]

        fig1 = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Portfolio Value vs Invested", "Units Accumulated"),
            horizontal_spacing=0.08,
        )
        # Portfolio value
        fig1.add_trace(go.Scatter(
            x=fp["date"], y=fp["portfolio_value"],
            name="Portfolio Value", line=dict(color=C["green"], width=2.5),
            fill="tozeroy", fillcolor="rgba(39,174,96,0.10)",
        ), row=1, col=1)
        fig1.add_trace(go.Scatter(
            x=fp["date"], y=fp["cum_invested"],
            name="Amount Invested", line=dict(color=C["orange"], width=1.5, dash="dash"),
        ), row=1, col=1)
        # Units
        fig1.add_trace(go.Scatter(
            x=sipd["date"], y=sipd["cum_units"],
            name="Cumulative Units", line=dict(color=C["purple"], width=2),
            fill="tozeroy", fillcolor="rgba(142,68,173,0.10)",
        ), row=1, col=2)
        fig1.update_layout(
            height=380, hovermode="x unified",
            legend=dict(orientation="h", y=1.1, x=0, font_size=10),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=60, b=40, l=50, r=50),
        )
        fig1.update_xaxes(showgrid=False)
        fig1.update_yaxes(showgrid=True, gridcolor="#ECF0F1")
        st.plotly_chart(fig1, use_container_width=True)

    # ── Second row: Donut + Monthly returns ───────────────────────────────────
    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown('<div class="section-header">Return Breakdown</div>',
                    unsafe_allow_html=True)
        donut = go.Figure(go.Pie(
            labels=["Invested", "Gains" if abs_ret >= 0 else "Loss"],
            values=[invested, max(abs_ret, 0)],
            hole=0.55,
            marker_colors=[C["blue"], C["green"] if abs_ret >= 0 else C["red"]],
            textinfo="label+percent",
            textfont_size=11,
        ))
        donut.update_layout(
            height=280, margin=dict(t=20, b=20, l=10, r=10),
            showlegend=False, paper_bgcolor="white",
            annotations=[dict(
                text=f"<b>{fmt_inr(end_val)}</b><br>Total", x=0.5, y=0.5,
                showarrow=False, font_size=11, align="center",
            )],
        )
        st.plotly_chart(donut, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Monthly NAV Returns (Investment Period)</div>',
                    unsafe_allow_html=True)
        if inv_type == "Lumpsum":
            pf_m = result["period_df"].set_index("date").resample("ME")["nav"].last()
        else:
            pf_m = result["full_period"].set_index("date").resample("ME")["nav"].last()

        nav_monthly_ret = pf_m.pct_change().dropna() * 100
        bar_clrs = [C["green"] if v >= 0 else C["red"] for v in nav_monthly_ret]
        bar_fig = go.Figure(go.Bar(
            x=nav_monthly_ret.index, y=nav_monthly_ret.values,
            marker_color=bar_clrs, name="Monthly NAV Return",
        ))
        bar_fig.update_layout(
            height=280, hovermode="x",
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10, b=30, l=40, r=20),
            yaxis=dict(title="Return %", gridcolor="#ECF0F1"),
            xaxis=dict(showgrid=False),
        )
        bar_fig.add_hline(y=0, line_color=C["gray"], line_width=0.8)
        st.plotly_chart(bar_fig, use_container_width=True)

    # ── NAV statistics ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">NAV Statistics (Investment Period)</div>',
                unsafe_allow_html=True)

    if inv_type == "Lumpsum":
        period_navs = result["period_df"]["nav"]
    else:
        period_navs = result["full_period"]["nav"]

    s1, s2, s3, s4, s5, s6 = st.columns(6)
    stats = [
        ("NAV at Entry",  f"₹{result['start_nav']:.4f}", C["blue"]),
        ("NAV Today",     f"₹{result['end_nav']:.4f}",   C["teal"]),
        ("Period High",   f"₹{period_navs.max():.4f}",   C["green"]),
        ("Period Low",    f"₹{period_navs.min():.4f}",   C["red"]),
        ("Std Dev (NAV)", f"₹{period_navs.std():.4f}",   C["purple"]),
        ("Vol (monthly)", f"{nav_monthly_ret.std():.2f}%", C["orange"]),
    ]
    for col, (lbl, val, clr) in zip([s1, s2, s3, s4, s5, s6], stats):
        with col:
            st.markdown(
                f'<div class="metric-card" style="border-left-color:{clr}">'
                f'<div class="metric-label">{lbl}</div>'
                f'<div class="metric-value" style="color:{clr};font-size:15px">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── SIP instalment table ───────────────────────────────────────────────────
    if inv_type == "SIP (Monthly)":
        with st.expander("📋 SIP Instalment Details", expanded=False):
            sipd = result["sip_df"].copy()
            sipd["date"] = sipd["date"].dt.strftime("%d %b %Y")
            sipd["nav"]  = sipd["nav"].apply(lambda x: f"₹{x:.4f}")
            sipd["units_bought"]  = sipd["units_bought"].apply(lambda x: f"{x:.4f}")
            sipd["cum_units"]     = sipd["cum_units"].apply(lambda x: f"{x:.4f}")
            sipd["sip_amount"]    = sipd["sip_amount"].apply(lambda x: fmt_inr_full(x))
            sipd["cum_invested"]  = sipd["cum_invested"].apply(lambda x: fmt_inr_full(x))
            sipd.columns = ["Date", "NAV", "Units Bought", "Cum. Units",
                            "SIP Amount", "Cum. Invested"]
            st.dataframe(sipd, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ██████  FUTURE PROJECTION MODE
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown('<div class="section-header">🔮 Future Projection — Return Simulator</div>',
                unsafe_allow_html=True)

    # Return rate picker — shown here since we need hist_ret
    with st.sidebar:
        st.markdown("### 📈 Return Rate")
        rate_options = {}
        for k, v in hist_ret.items():
            rate_options[f"{k} CAGR ({v:.2f}%)"] = v
        rate_options["Custom %"] = None

        rate_label = st.selectbox(
            "Use return rate from",
            list(rate_options.keys()),
            help="Pick a historical CAGR to use as projection base, or enter custom",
        )
        if rate_options[rate_label] is None:
            custom_rate = st.number_input(
                "Custom annual return %", min_value=0.1, max_value=50.0,
                value=12.0, step=0.5,
            )
            base_rate = custom_rate
        else:
            base_rate = rate_options[rate_label]

        st.caption(f"Base rate: **{base_rate:.2f}% p.a.**")

        inflation = st.slider("Inflation rate (% for adj.)", 0, 15, 6,
                              help="Used to compute inflation-adjusted final value")

    # Three scenarios
    bear_rate = max(base_rate * 0.55, 1.0)
    bull_rate = base_rate * 1.45

    inv_label = "Lumpsum" if inv_type == "Lumpsum" else "SIP (Monthly)"

    bear_df = project_future(amount, inv_label, bear_rate, proj_years, stepup)
    base_df = project_future(amount, inv_label, base_rate, proj_years, stepup)
    bull_df = project_future(amount, inv_label, bull_rate, proj_years, stepup)

    invested_f = base_df["invested"].iloc[-1]
    end_bear   = bear_df["value"].iloc[-1]
    end_base   = base_df["value"].iloc[-1]
    end_bull   = bull_df["value"].iloc[-1]
    infl_adj   = end_base / ((1 + inflation / 100) ** proj_years)

    # ── KPI row ───────────────────────────────────────────────────────────────
    kpi = st.columns(5)
    kpis = [
        ("Total Invested",    fmt_inr(invested_f),  f"{proj_years} yrs · {inv_label}",                         C["blue"]),
        ("Bear Case (Est.)",  fmt_inr(end_bear),    f"{bear_rate:.1f}% p.a. · gain: {fmt_inr(end_bear-invested_f)}", C["orange"]),
        ("Base Case",         fmt_inr(end_base),    f"{base_rate:.1f}% p.a. · gain: {fmt_inr(end_base-invested_f)}", C["green"]),
        ("Bull Case (Est.)",  fmt_inr(end_bull),    f"{bull_rate:.1f}% p.a. · gain: {fmt_inr(end_bull-invested_f)}", C["purple"]),
        ("Inflation-Adj.",    fmt_inr(infl_adj),    f"@ {inflation}% inflation over {proj_years}y",                  C["gold"]),
    ]
    for col, (lbl, val, sub, clr) in zip(kpi, kpis):
        with col:
            st.markdown(metric_card(lbl, val, sub, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Growth Chart ─────────────────────────────────────────────────────────
    x_vals = base_df["year"]

    fig_proj = go.Figure()

    # Bull band
    fig_proj.add_trace(go.Scatter(
        x=list(x_vals) + list(x_vals[::-1]),
        y=list(bull_df["value"]) + list(bear_df["value"][::-1]),
        fill="toself",
        fillcolor="rgba(39,174,96,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Bear–Bull Range", showlegend=True,
        hoverinfo="skip",
    ))

    fig_proj.add_trace(go.Scatter(
        x=x_vals, y=bear_df["value"],
        name=f"Bear ({bear_rate:.1f}%)",
        line=dict(color=C["orange"], width=1.5, dash="dot"),
    ))
    fig_proj.add_trace(go.Scatter(
        x=x_vals, y=base_df["value"],
        name=f"Base ({base_rate:.1f}%)",
        line=dict(color=C["green"], width=2.5),
        fill="tonexty" if False else None,
    ))
    fig_proj.add_trace(go.Scatter(
        x=x_vals, y=bull_df["value"],
        name=f"Bull ({bull_rate:.1f}%)",
        line=dict(color=C["purple"], width=1.5, dash="dash"),
    ))
    fig_proj.add_trace(go.Scatter(
        x=x_vals, y=base_df["invested"],
        name="Invested Capital",
        line=dict(color=C["blue"], width=2, dash="dashdot"),
    ))
    # Inflation-adjusted baseline
    infl_vals = [end_base / ((1 + inflation / 100) ** (proj_years - yr)) for yr in x_vals]
    fig_proj.add_trace(go.Scatter(
        x=x_vals, y=infl_vals,
        name=f"Infl-adj ({inflation}%)",
        line=dict(color=C["gray"], width=1, dash="dot"),
        opacity=0.6,
    ))

    fig_proj.update_layout(
        title=dict(
            text=f"Projected Growth · {proj_years}yr · {inv_label} · {selected_name[:50]}",
            font_size=13,
        ),
        height=420, hovermode="x unified",
        xaxis=dict(title="Years", showgrid=False),
        yaxis=dict(title="Value (₹)", showgrid=True, gridcolor="#ECF0F1"),
        legend=dict(orientation="h", y=1.08, x=0, font_size=10),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=60, b=40, l=60, r=30),
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    # ── Second row: Donut + Year-by-year milestone table ──────────────────────
    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown('<div class="section-header">Invested vs Gains (Base Case)</div>',
                    unsafe_allow_html=True)
        base_gains = end_base - invested_f
        donut2 = go.Figure(go.Pie(
            labels=["Invested", "Estimated Gains"],
            values=[invested_f, max(base_gains, 0)],
            hole=0.55,
            marker_colors=[C["blue"], C["green"]],
            textinfo="label+percent",
            textfont_size=11,
        ))
        donut2.update_layout(
            height=300, margin=dict(t=20, b=20, l=10, r=10),
            showlegend=False, paper_bgcolor="white",
            annotations=[dict(
                text=f"<b>{fmt_inr(end_base)}</b><br>at {base_rate:.1f}%",
                x=0.5, y=0.5, showarrow=False, font_size=11, align="center",
            )],
        )
        st.plotly_chart(donut2, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Year-by-Year Milestones (Base Case)</div>',
                    unsafe_allow_html=True)
        milestones = []
        for yr in range(1, proj_years + 1):
            row_b = bear_df[bear_df["month"] == yr * 12].iloc[0]
            row_m = base_df[base_df["month"] == yr * 12].iloc[0]
            row_u = bull_df[bull_df["month"] == yr * 12].iloc[0]
            milestones.append({
                "Year": yr,
                "Invested": fmt_inr_full(row_m["invested"]),
                f"Bear ({bear_rate:.1f}%)": fmt_inr(row_b["value"]),
                f"Base ({base_rate:.1f}%)": fmt_inr(row_m["value"]),
                f"Bull ({bull_rate:.1f}%)": fmt_inr(row_u["value"]),
                "Gain (Base)": f"{(row_m['value']-row_m['invested'])/row_m['invested']*100:.1f}%" if row_m["invested"] > 0 else "—",
            })
        st.dataframe(pd.DataFrame(milestones), use_container_width=True, hide_index=True)

    # ── Compare using different historical CAGR rates ─────────────────────────
    if hist_ret:
        st.markdown('<div class="section-header">Scenario Comparison (All Historical CAGRs)</div>',
                    unsafe_allow_html=True)

        cmp_rows = []
        for period, rate in hist_ret.items():
            df_tmp   = project_future(amount, inv_label, rate, proj_years, stepup)
            end_tmp  = df_tmp["value"].iloc[-1]
            inv_tmp  = df_tmp["invested"].iloc[-1]
            cmp_rows.append({
                "Period Used": period,
                "CAGR (%)": f"{rate:.2f}%",
                "Invested": fmt_inr_full(inv_tmp),
                "Projected Value": fmt_inr(end_tmp),
                "Total Gain": fmt_inr(end_tmp - inv_tmp),
                "Absolute Return %": f"{(end_tmp-inv_tmp)/inv_tmp*100:.1f}%",
                "Infl-adj Value": fmt_inr(end_tmp / ((1+inflation/100)**proj_years)),
            })
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    # ── Projection assumptions note ───────────────────────────────────────────
    st.markdown(f"""
    <div class="info-strip">
    <b>⚠️ Projection Disclaimer:</b> These are estimates based on historical CAGR from live mfapi.in NAV data.
    Mutual fund returns are subject to market risk. Past performance is not indicative of future results.
    Bear / Bull scenarios use {bear_rate:.1f}% and {bull_rate:.1f}% respectively (55% and 145% of base rate).
    Inflation adjustment uses {inflation}% p.a.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER — Full NAV History Chart
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📈 Full NAV History Chart", expanded=False):
    fig_full = go.Figure()
    fig_full.add_trace(go.Scatter(
        x=navs["date"], y=navs["nav"],
        line=dict(color=C["blue"], width=1.5),
        name="NAV", fill="tozeroy", fillcolor="rgba(27,82,153,0.07)",
    ))
    fig_full.update_layout(
        title=f"Full NAV History · {selected_name[:60]}",
        height=350, hovermode="x",
        xaxis=dict(showgrid=False, rangeslider=dict(visible=True)),
        yaxis=dict(title="NAV (₹)", showgrid=True, gridcolor="#ECF0F1"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=50, b=40, l=60, r=20),
    )
    st.plotly_chart(fig_full, use_container_width=True)

with st.expander("📋 Raw NAV Data (last 90 days)", expanded=False):
    tail = navs.tail(90).copy()
    tail["date"] = tail["date"].dt.strftime("%d %b %Y")
    tail["nav"]  = tail["nav"].apply(lambda x: f"₹{x:.4f}")
    tail.columns = ["Date", "NAV"]
    st.dataframe(tail[::-1], use_container_width=True, hide_index=True)

st.caption(
    f"Data source: [mfapi.in](https://www.mfapi.in) — free, open mutual fund NAV API · "
    f"Refreshed every 5 min · Dashboard: {TODAY.strftime('%d %b %Y')}"
)
