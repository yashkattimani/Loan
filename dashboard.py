#!/usr/bin/env python3
"""
Education Loan Analytics Dashboard
— Login-protected, manual payment tracking, prepayment calculator
Run:  streamlit run dashboard.py
"""

import hashlib
import json
import os
import re
import time
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD CONFIG FROM secrets.toml  (safe fallbacks for local dev)
# ─────────────────────────────────────────────────────────────────────────────
def _secret(section: str, key: str, default=""):
    try:
        return st.secrets[section][key]
    except Exception:
        return default

AUTH_USER   = _secret("auth", "username",       "admin")
AUTH_PASS   = _secret("auth", "password",       "changeme")
LOAN_ACCT   = _secret("loan", "account_number", "XXXXXXXXXXXXXXX")
BANK        = _secret("loan", "bank_name",      "Union Bank of India")
RATE_PA     = float(_secret("loan", "rate_pa",   8.55)) / 100
CSV_FILE    = _secret("loan", "csv_file",        "Loan analyzer.csv")
MANUAL_FILE = "manual_payments.json"

MASKED_ACCT = f"****{LOAN_ACCT[-4:]}" if len(LOAN_ACCT) >= 4 else "****"

COLORS = dict(
    navy="#1A2E4A", blue="#1B5299", gold="#C8912A",
    green="#27AE60", red="#C0392B", orange="#E67E22",
    purple="#8E44AD", teal="#16A085", gray="#7F8C8D",
    bg="#F5F7FA", d1="#3498DB", d2="#2ECC71", d3="#E74C3C", d4="#F39C12",
)
TRANCHE_COLS = [COLORS["d1"], COLORS["d2"], COLORS["d3"], COLORS["d4"]]

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
  .metric-card {
    background: white; border-radius: 10px; padding: 12px 16px;
    border-left: 4px solid; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    height: 100%; margin-bottom: 8px;
  }
  .metric-label { font-size: 10px; font-weight: 700; color: #7F8C8D;
                  letter-spacing: 0.6px; text-transform: uppercase; }
  .metric-value { font-size: 18px; font-weight: 700; margin: 5px 0 3px; }
  .metric-sub   { font-size: 10px; color: #95A5A6; line-height: 1.4; }
  .section-header {
    font-size: 13px; font-weight: 700; color: #1A2E4A;
    border-bottom: 2px solid #C8912A; padding-bottom: 4px;
    margin-bottom: 12px; margin-top: 4px;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
  .stTabs [data-baseweb="tab"] {
    height: 38px; padding: 0 14px;
    background: white; border-radius: 6px 6px 0 0;
    font-weight: 600; font-size: 12px;
  }
  .stTabs [aria-selected="true"] {
    background: #1A2E4A !important; color: white !important;
  }
  .savings-box {
    background: linear-gradient(135deg, #27AE60, #2ECC71);
    border-radius: 10px; padding: 16px 20px; color: white; text-align: center;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def show_login():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.markdown(f"""
        <div style="background:#1A2E4A;padding:32px;border-radius:16px;
                    text-align:center;margin-bottom:24px;
                    box-shadow:0 8px 32px rgba(0,0,0,0.18);">
          <div style="font-size:48px;">🏦</div>
          <div style="color:#C8912A;font-size:15px;font-weight:700;
                      letter-spacing:2px;margin-top:10px;">LOAN ANALYTICS</div>
          <div style="color:#BDC3C7;font-size:11px;margin-top:6px;">
            Private Dashboard — Sign In to Continue
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password",
                                     placeholder="Enter password")
            submitted = st.form_submit_button("🔐  Sign In",
                                              use_container_width=True)

        if submitted:
            if username == AUTH_USER and _hash(password) == _hash(AUTH_PASS):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown(
            "<p style='text-align:center;color:#95A5A6;font-size:10px;"
            "margin-top:16px;'>For personal use only</p>",
            unsafe_allow_html=True,
        )

if not st.session_state.get("authenticated", False):
    show_login()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_inr(x: float) -> str:
    ax = abs(x)
    if ax >= 1e7:  return f"₹{x/1e7:.2f} Cr"
    if ax >= 1e5:  return f"₹{x/1e5:.2f} L"
    if ax >= 1e3:  return f"₹{x/1e3:.1f}K"
    return f"₹{x:.0f}"

def fmt_inr_full(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"₹ {sign}{abs(x):,.0f}"

def parse_amt(s) -> float:
    if pd.isna(s) or not str(s).strip(): return 0.0
    s  = str(s).strip()
    dr = "(Dr)" in s
    val = float(re.sub(r"[^\d.]", "", s.replace("(Dr)", "").replace("(Cr)", "")))
    return -val if dr else val

def parse_bal(s) -> float:
    if pd.isna(s) or not str(s).strip(): return np.nan
    try:    return float(re.sub(r"[,\s]", "", str(s).strip()))
    except: return np.nan

def months_str(df_) -> str:
    if df_["bal"].iloc[-1] > 100: return "Never (underpays)"
    y, m = divmod(len(df_), 12)
    return (f"{y}y {m}m" if y else f"{m}m")

def project(monthly_pmt: float, outstanding: float, max_months: int = 360) -> pd.DataFrame:
    r, bal, rows = RATE_PA / 12, outstanding, []
    for mo in range(1, max_months + 1):
        interest  = bal * r
        principal = max(0.0, monthly_pmt - interest)
        bal      -= principal
        rows.append(dict(month=mo, bal=max(0.0, bal),
                         interest=interest, principal=principal))
        if bal <= 0:
            break
    return pd.DataFrame(rows)

def project_with_lumpsum(monthly_pmt: float, outstanding: float,
                         lumpsum: float, lumpsum_month: int,
                         max_months: int = 360) -> pd.DataFrame:
    r, bal, rows = RATE_PA / 12, outstanding, []
    for mo in range(1, max_months + 1):
        if mo == lumpsum_month:
            bal = max(0.0, bal - lumpsum)
        interest  = bal * r
        principal = max(0.0, monthly_pmt - interest)
        bal       = max(0.0, bal - principal)
        rows.append(dict(month=mo, bal=bal, interest=interest, principal=principal))
        if bal <= 0:
            break
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
#  MANUAL PAYMENTS  —  read / write JSON
# ─────────────────────────────────────────────────────────────────────────────
def load_manual_payments() -> list:
    if os.path.exists(MANUAL_FILE):
        try:
            with open(MANUAL_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_manual_payment(pdate, amount: float, remarks: str):
    payments = load_manual_payments()
    payments.append({
        "date":    str(pdate),
        "amount":  float(amount),
        "remarks": str(remarks) if remarks else "Manual Payment",
    })
    with open(MANUAL_FILE, "w") as f:
        json.dump(payments, f, indent=2)

def delete_manual_payment(idx: int):
    payments = load_manual_payments()
    if 0 <= idx < len(payments):
        payments.pop(idx)
        with open(MANUAL_FILE, "w") as f:
            json.dump(payments, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data(manual_version: int = 0) -> pd.DataFrame:
    raw = pd.read_csv(CSV_FILE)
    raw.columns = ["Date", "TranId", "Remarks", "Amount", "Balance",
                   "Deposited", "BankInt"]
    raw["Date"]   = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
    raw["Amt"]    = raw["Amount"].apply(parse_amt)
    raw["Bal"]    = raw["Balance"].apply(parse_bal)
    raw           = raw.sort_values("Date").reset_index(drop=True)
    raw["Abs"]    = raw["Amt"].abs()
    raw["Source"] = "csv"

    def classify(r):
        rem = str(r["Remarks"]).lower()
        v   = r["Amt"]
        if "loan disbursement" in rem and v < 0: return "Disbursement"
        if re.search(r":n int\.:", rem):         return "Interest"
        if "loan coll" in rem and v > 0:         return "IntPmt"
        if v > 0:                                return "Payment"
        return "Other"

    raw["Type"] = raw.apply(classify, axis=1)

    manual = load_manual_payments()
    if manual:
        m_rows = []
        for p in manual:
            m_rows.append({
                "Date":     pd.Timestamp(p["date"]),
                "TranId":   "MANUAL",
                "Remarks":  p.get("remarks", "Manual Payment"),
                "Amount":   f"{p['amount']:.2f}(Cr)",
                "Balance":  None,
                "Deposited": None,
                "BankInt":  None,
                "Amt":      float(p["amount"]),
                "Bal":      np.nan,
                "Abs":      float(p["amount"]),
                "Source":   "manual",
                "Type":     "Payment",
            })
        raw = pd.concat([raw, pd.DataFrame(m_rows)], ignore_index=True)
        raw = raw.sort_values("Date").reset_index(drop=True)

    return raw


@st.cache_data(ttl=300)
def compute_metrics(df_json: str, today_str: str, manual_version: int = 0) -> dict:
    df     = pd.read_json(df_json, orient="records")
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    if "Source" not in df.columns:
        df["Source"] = "csv"
    TODAY  = datetime.strptime(today_str, "%Y-%m-%d").date()

    disb   = df[df["Type"] == "Disbursement"].reset_index(drop=True)
    ints   = df[df["Type"] == "Interest"].reset_index(drop=True)
    pmts   = df[df["Type"] == "Payment"].reset_index(drop=True)
    ipmts  = df[df["Type"] == "IntPmt"].reset_index(drop=True)

    total_disbursed   = disb["Abs"].sum()
    total_int_charged = ints["Abs"].sum()
    total_paid_pmts   = pmts["Abs"].sum()
    total_paid_ipmts  = ipmts["Abs"].sum()
    total_repaid      = total_paid_pmts + total_paid_ipmts   # every rupee paid in

    # Outstanding: last CSV balance minus any manual payments after it
    csv_df   = df[df["Source"] == "csv"]
    last_csv = csv_df[csv_df["Bal"].notna()].sort_values("Date")
    if len(last_csv) == 0:
        outstanding = total_disbursed
        last_csv_date = pd.Timestamp(TODAY)
    else:
        last_csv_row  = last_csv.iloc[-1]
        last_csv_date = last_csv_row["Date"]
        csv_outstanding = abs(last_csv_row["Bal"])
        manual_after    = df[
            (df["Source"] == "manual") & (df["Date"] > last_csv_date)
        ]["Abs"].sum()
        outstanding = max(0.0, csv_outstanding - manual_after)

    principal_repaid  = max(0.0, total_disbursed - outstanding)
    loan_start        = disb["Date"].min()
    if hasattr(loan_start, "tzinfo") and loan_start.tzinfo is not None:
        loan_start = loan_start.tz_localize(None)
    loan_age_days   = (pd.Timestamp(TODAY) - loan_start).days
    loan_age_months = loan_age_days / 30.44

    # Balance time series: CSV rows + synthetic extension for manual payments
    bal_ts         = csv_df[csv_df["Bal"].notna()][["Date", "Bal"]].copy()
    bal_ts["Outs"] = bal_ts["Bal"].abs()
    bal_ts         = bal_ts.sort_values("Date").reset_index(drop=True)

    if len(last_csv) > 0:
        running     = float(last_csv.iloc[-1]["Bal"])   # negative value
        lc_date     = last_csv.iloc[-1]["Date"]
        manual_ext  = df[
            (df["Source"] == "manual") & (df["Date"] > lc_date)
        ].sort_values("Date")
        for _, mp in manual_ext.iterrows():
            running = min(running + mp["Abs"], 0.0)
            bal_ts  = pd.concat([bal_ts, pd.DataFrame([{
                "Date": mp["Date"],
                "Bal":  running,
                "Outs": abs(running),
            }])], ignore_index=True)
        bal_ts = bal_ts.sort_values("Date").reset_index(drop=True)

    ints_s           = ints.sort_values("Date").copy()
    ints_s["Month"]  = ints_s["Date"].dt.to_period("M").astype(str)
    mon_int          = ints_s.groupby("Month")["Abs"].sum().reset_index()
    mon_int["Dt"]    = pd.to_datetime(mon_int["Month"])
    ints_s["CumInt"] = ints_s["Abs"].cumsum()

    all_pmts            = df[df["Type"].isin(["Payment", "IntPmt"])].sort_values("Date").copy()
    all_pmts["Month"]   = all_pmts["Date"].dt.to_period("M").astype(str)
    mon_pmt             = all_pmts.groupby("Month")["Abs"].sum().reset_index()
    mon_pmt["Dt"]       = pd.to_datetime(mon_pmt["Month"])

    pmts_s           = pmts.sort_values("Date").copy()
    pmts_s["Month"]  = pmts_s["Date"].dt.to_period("M").astype(str)
    pmts_s["CumPmt"] = pmts_s["Abs"].cumsum()

    return dict(
        disb=disb.to_dict("records"),
        ints=ints.to_dict("records"),
        pmts=pmts.to_dict("records"),
        ipmts=ipmts.to_dict("records"),
        ints_s=ints_s.to_dict("records"),
        pmts_s=pmts_s.to_dict("records"),
        total_disbursed=total_disbursed,
        total_int_charged=total_int_charged,
        total_paid_pmts=total_paid_pmts,
        total_paid_ipmts=total_paid_ipmts,
        total_repaid=total_repaid,
        outstanding=outstanding,
        principal_repaid=principal_repaid,
        loan_start=loan_start.isoformat(),
        loan_age_days=loan_age_days,
        loan_age_months=loan_age_months,
        bal_ts=bal_ts.to_dict("records"),
        mon_int=mon_int.to_dict("records"),
        mon_pmt=mon_pmt.to_dict("records"),
        n_ints=len(ints),
    )


@st.cache_data(ttl=300)
def calc_theory(bal_ts_json: str, today_str: str) -> pd.DataFrame:
    bal_ts     = pd.read_json(bal_ts_json, orient="records")
    bal_ts["Date"] = pd.to_datetime(bal_ts["Date"], utc=True).dt.tz_localize(None)
    TODAY      = datetime.strptime(today_str, "%Y-%m-%d").date()
    daily_rate = RATE_PA / 365
    dates      = pd.date_range(bal_ts["Date"].min(), pd.Timestamp(TODAY), freq="D")
    bal_lkp    = bal_ts.set_index("Date")["Outs"]
    rows, cum  = [], 0.0
    for d in dates:
        known = bal_lkp[bal_lkp.index <= d]
        bal   = float(known.iloc[-1]) if len(known) else 0.0
        di    = bal * daily_rate
        cum  += di
        rows.append({"Date": d, "Bal": bal, "DailyInt": di, "CumInt": cum})
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def calc_implied_rates(ints_s_json: str, bal_ts_json: str) -> pd.DataFrame:
    ints_s = pd.read_json(ints_s_json, orient="records")
    ints_s["Date"] = pd.to_datetime(ints_s["Date"], utc=True).dt.tz_localize(None)
    bal_ts = pd.read_json(bal_ts_json, orient="records")
    bal_ts["Date"] = pd.to_datetime(bal_ts["Date"], utc=True).dt.tz_localize(None)

    def parse_int_period(rem):
        m = re.search(r"(\d{2}-\d{2}-\d{4})\s+to\s+(\d{2}-\d{2}-\d{4})", str(rem))
        if m:
            s = datetime.strptime(m.group(1), "%d-%m-%Y")
            e = datetime.strptime(m.group(2), "%d-%m-%Y")
            return s, e, (e - s).days + 1
        return None, None, None

    parsed = pd.DataFrame(
        ints_s["Remarks"].apply(parse_int_period).tolist(),
        columns=["P_Start", "P_End", "P_Days"], index=ints_s.index,
    )
    ints_s = pd.concat([ints_s, parsed], axis=1)

    implied_rates = []
    for _, row in ints_s.iterrows():
        if row["P_Start"] and row["P_Days"]:
            prior = bal_ts[bal_ts["Date"] <= pd.Timestamp(row["P_Start"])]["Outs"]
            if len(prior):
                b = float(prior.iloc[-1])
                if b > 0 and row["P_Days"] > 0:
                    r = row["Abs"] / b * 365 / row["P_Days"] * 100
                    implied_rates.append({"Date": row["Date"], "Rate": r})

    return (pd.DataFrame(implied_rates) if implied_rates
            else pd.DataFrame(columns=["Date", "Rate"]))

# ─────────────────────────────────────────────────────────────────────────────
#  INITIALISE SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "manual_version" not in st.session_state:
    st.session_state.manual_version = 0
if "default_emi" not in st.session_state:
    st.session_state.default_emi = 20_000
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏦 Loan Dashboard"

TODAY         = date.today()
today_str     = TODAY.strftime("%Y-%m-%d")
_on_loan_page = (st.session_state.get("current_page", "🏦 Loan Dashboard") == "🏦 Loan Dashboard")

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD & COMPUTE  (only when on loan page)
# ─────────────────────────────────────────────────────────────────────────────
if _on_loan_page:
    mv      = st.session_state.manual_version
    raw_df  = load_data(manual_version=mv)
    df_json = raw_df.to_json(orient="records", date_format="iso")
    m       = compute_metrics(df_json, today_str, manual_version=mv)

    def _dt(df, col="Date"):
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
        return df

    disb              = _dt(pd.DataFrame(m["disb"]))
    ints_df           = _dt(pd.DataFrame(m["ints"]))
    pmts_df           = _dt(pd.DataFrame(m["pmts"]))
    ipmts_df          = _dt(pd.DataFrame(m["ipmts"]))
    ints_s            = _dt(pd.DataFrame(m["ints_s"]))
    pmts_s            = _dt(pd.DataFrame(m["pmts_s"]))
    bal_ts            = _dt(pd.DataFrame(m["bal_ts"]))
    mon_int           = pd.DataFrame(m["mon_int"]); mon_int["Dt"] = pd.to_datetime(mon_int["Dt"])
    mon_pmt           = pd.DataFrame(m["mon_pmt"]); mon_pmt["Dt"] = pd.to_datetime(mon_pmt["Dt"])

    total_disbursed   = m["total_disbursed"]
    total_int_charged = m["total_int_charged"]
    total_repaid      = m["total_repaid"]
    outstanding       = m["outstanding"]
    principal_repaid  = m["principal_repaid"]
    loan_start        = pd.Timestamp(m["loan_start"])
    loan_age_days     = m["loan_age_days"]
    loan_age_months   = m["loan_age_months"]
    n_ints            = m["n_ints"]

    bal_ts_json  = bal_ts.to_json(orient="records", date_format="iso")
    ints_s_json  = ints_s[["Date", "Remarks", "Abs"]].to_json(orient="records", date_format="iso")

    theory           = calc_theory(bal_ts_json, today_str)
    rate_df          = calc_implied_rates(ints_s_json, bal_ts_json)
    theory_total_int = theory["CumInt"].iloc[-1]

    disb_meta = []
    for i in range(len(disb)):
        row = disb.iloc[i]
        dt  = row["Date"].date()
        amt = row["Abs"]
        de  = (TODAY - dt).days
        disb_meta.append(dict(
            n=i+1, tag=row["Date"].strftime("%b'%y"),
            dt=dt, amt=amt, col=TRANCHE_COLS[i % 4],
            days=de, theory_int=amt * RATE_PA / 365 * de,
        ))

    breakeven = outstanding * RATE_PA / 12

# ─────────────────────────────────────────────────────────────────────────────
#  MUTUAL FUNDS — UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
_MF_KNOWN_AMCS = [
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

def _mf_extract_amc(name: str) -> str:
    for amc in sorted(_MF_KNOWN_AMCS, key=len, reverse=True):
        if name.lower().startswith(amc.lower()):
            return amc
    parts = name.split()
    return " ".join(parts[:min(2, len(parts))])

def _mf_cagr(start_val: float, end_val: float, years: float) -> float:
    if years <= 0 or start_val <= 0 or end_val <= 0:
        return 0.0
    return ((end_val / start_val) ** (1.0 / years) - 1) * 100

def _mf_card(label: str, value: str, sub: str, color: str) -> str:
    return (
        f'<div class="metric-card" style="border-left-color:{color}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color}">{value}</div>'
        f'<div class="metric-sub">{sub}</div>'
        f'</div>'
    )

@st.cache_data(ttl=3600, show_spinner=False)
def _mf_fetch_list() -> list:
    r = requests.get("https://api.mfapi.in/mf", timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300, show_spinner=False)
def _mf_fetch_nav(code: int) -> tuple:
    r = requests.get(f"https://api.mfapi.in/mf/{code}", timeout=20)
    r.raise_for_status()
    data = r.json()
    meta = data.get("meta", {})
    df   = pd.DataFrame(data.get("data", []))
    if df.empty:
        return meta, pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["nav"]).sort_values("date").reset_index(drop=True)
    return meta, df

@st.cache_data(ttl=3600, show_spinner=False)
def _mf_build_df(raw_json: str) -> pd.DataFrame:
    fund_list = json.loads(raw_json)
    df = pd.DataFrame(fund_list)
    df = df.rename(columns={"schemeCode": "code", "schemeName": "name"})
    # Keep only active funds (have a live ISIN) and deduplicate by scheme code
    if "isinGrowth" in df.columns:
        df = df[df["isinGrowth"].notna()]
    df = df.drop_duplicates(subset=["code"])
    df = df[["code", "name"]].dropna(subset=["name"])
    df["name"] = df["name"].astype(str)
    df["amc"] = df["name"].apply(_mf_extract_amc)
    return df

def _mf_nav_at(navs: pd.DataFrame, target: date):
    ts  = pd.Timestamp(target)
    fwd = navs[navs["date"] >= ts]
    if not fwd.empty:
        return fwd.iloc[0]
    bwd = navs[navs["date"] < ts]
    return bwd.iloc[-1] if not bwd.empty else None

def _mf_lumpsum(navs: pd.DataFrame, start: date, amount: float) -> dict | None:
    row0 = _mf_nav_at(navs, start)
    if row0 is None:
        return None
    start_nav = row0["nav"]
    start_dt  = row0["date"]
    units     = amount / start_nav
    period    = navs[navs["date"] >= start_dt].copy()
    period["portfolio_value"] = units * period["nav"]
    period["invested"]        = amount
    period["gain"]            = period["portfolio_value"] - amount
    end_val = period["portfolio_value"].iloc[-1]
    end_dt  = period["date"].iloc[-1]
    years   = max((end_dt - start_dt).days / 365.25, 0.001)
    return dict(
        inv_type="Lumpsum", invested=amount, units=units,
        start_nav=start_nav, end_nav=navs["nav"].iloc[-1],
        end_value=end_val, abs_return=end_val - amount,
        abs_pct=(end_val - amount) / amount * 100,
        cagr=_mf_cagr(amount, end_val, years),
        years=years, period_df=period,
        start_date=start_dt.date(), end_date=end_dt.date(),
    )

def _mf_sip(navs: pd.DataFrame, start: date, monthly: float,
            end: date | None = None, stepup_pct: float = 0.0) -> dict | None:
    if end is None:
        end = navs["date"].iloc[-1].date()
    rows, total_units, total_invested = [], 0.0, 0.0
    sip_amt = monthly
    cur     = date(start.year, start.month, 1)
    while cur <= end:
        row = _mf_nav_at(navs, cur)
        if row is not None:
            nav   = row["nav"]
            units = sip_amt / nav
            total_units    += units
            total_invested += sip_amt
            rows.append(dict(date=row["date"], nav=nav, units_bought=units,
                             cum_units=total_units, cum_invested=total_invested,
                             sip_amount=sip_amt))
        nxt = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
        if stepup_pct > 0 and nxt.month == 1 and nxt != date(start.year, 1, 1):
            sip_amt *= (1 + stepup_pct / 100)
        cur = nxt
    if not rows:
        return None
    sip_df  = pd.DataFrame(rows)
    full    = navs[navs["date"] >= sip_df["date"].min()].copy().reset_index(drop=True)
    cum_u, cum_i = [], []
    for d in full["date"]:
        prior = sip_df[sip_df["date"] <= d]
        cum_u.append(prior["cum_units"].iloc[-1] if not prior.empty else 0.0)
        cum_i.append(prior["cum_invested"].iloc[-1] if not prior.empty else 0.0)
    full["cum_units"]       = cum_u
    full["cum_invested"]    = cum_i
    full["portfolio_value"] = full["cum_units"] * full["nav"]
    full["gain"]            = full["portfolio_value"] - full["cum_invested"]
    end_val = total_units * navs["nav"].iloc[-1]
    abs_ret = end_val - total_invested
    years   = max((sip_df["date"].iloc[-1] - sip_df["date"].iloc[0]).days / 365.25, 0.001)
    approx_cagr = _mf_cagr(total_invested / 2, end_val, years / 2) if years > 0.5 else 0.0
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

def _mf_hist_returns(navs: pd.DataFrame) -> dict:
    if navs.empty:
        return {}
    latest_nav  = navs["nav"].iloc[-1]
    latest_date = navs["date"].iloc[-1]
    out = {}
    for yrs in [1, 2, 3, 5, 7, 10]:
        target = latest_date - pd.DateOffset(years=yrs)
        past   = navs[navs["date"] <= target]
        if not past.empty:
            out[f"{yrs}yr"] = round(_mf_cagr(past["nav"].iloc[-1], latest_nav, yrs), 2)
    return out

def _mf_project_future(amount: float, inv_type: str, rate_pct: float,
                       years: int, stepup_pct: float = 0.0) -> pd.DataFrame:
    rate_m = rate_pct / 100 / 12
    rows   = []
    if inv_type == "Lumpsum":
        for mo in range(years * 12 + 1):
            rows.append(dict(month=mo, year=mo / 12, invested=amount,
                             value=amount * (1 + rate_pct / 100) ** (mo / 12)))
    else:
        val, invested, sip = 0.0, 0.0, amount
        rows.append(dict(month=0, year=0, invested=0, value=0))
        for mo in range(1, years * 12 + 1):
            val      = (val + sip) * (1 + rate_m)
            invested += sip
            rows.append(dict(month=mo, year=mo / 12, invested=invested, value=val))
            if mo % 12 == 0 and stepup_pct > 0:
                sip *= (1 + stepup_pct / 100)
    df = pd.DataFrame(rows)
    df["gain"]     = df["value"] - df["invested"]
    df["gain_pct"] = df["gain"] / df["invested"].replace(0, np.nan) * 100
    return df

# ─────────────────────────────────────────────────────────────────────────────
#  MUTUAL FUNDS — FULL PAGE RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def render_mf_page():
    MFC = dict(
        navy="#1A2E4A", blue="#1B5299", gold="#C8912A",
        green="#27AE60", red="#C0392B", orange="#E67E22",
        purple="#8E44AD", teal="#16A085", gray="#7F8C8D",
    )

    # ── Load fund list ─────────────────────────────────────────────────────────
    with st.spinner("Loading active funds from mfapi.in…"):
        try:
            raw_funds = _mf_fetch_list()
            fund_df   = _mf_build_df(json.dumps(raw_funds))
        except Exception as e:
            st.error(f"mfapi.in unreachable: {e}")
            return

    # ── Sidebar controls ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"**{len(fund_df):,} funds loaded**")
        st.markdown("### 🔍 Find a Fund")

        search_mode = st.radio("Search by", ["AMC → Fund Name", "Direct Search"],
                               horizontal=True, key="mf_search_mode")

        selected_code, selected_name = None, None

        if search_mode == "AMC → Fund Name":
            amc_list   = ["— Select AMC —"] + sorted(fund_df["amc"].unique())
            chosen_amc = st.selectbox("Fund House / AMC", amc_list, key="mf_amc")
            if chosen_amc != "— Select AMC —":
                sub   = fund_df[fund_df["amc"] == chosen_amc].reset_index(drop=True)
                names = sub["name"].tolist()
                codes = sub["code"].tolist()
                fidx  = st.selectbox("Fund Name", range(len(names)),
                                     format_func=lambda i: names[i], key="mf_fund_idx")
                selected_code = int(codes[fidx])
                selected_name = names[fidx]
        else:
            term = st.text_input("Search fund name", key="mf_search_term",
                                 placeholder="e.g. HDFC Flexi Cap Growth")
            if term:
                hits = fund_df[fund_df["name"].str.contains(term, case=False, na=False)].reset_index(drop=True)
                if hits.empty:
                    st.warning("No matching funds found.")
                else:
                    names = hits["name"].tolist()
                    codes = hits["code"].tolist()
                    fidx  = st.selectbox("Select fund", range(len(names)),
                                         format_func=lambda i: names[i], key="mf_search_idx")
                    selected_code = int(codes[fidx])
                    selected_name = names[fidx]

        if selected_code:
            st.divider()
            st.markdown("### 🗂️ Mode")
            mf_mode = st.radio(
                "Choose mode",
                ["📅 Historical Backtest", "🔮 Future Projection"],
                key="mf_mode",
                help="Historical: real NAV data  |  Future: project with chosen rate",
            )
            st.divider()
            st.markdown("### 💰 Investment")
            mf_inv_type = st.radio("Type", ["Lumpsum", "SIP (Monthly)"],
                                   horizontal=True, key="mf_inv_type")
            if mf_inv_type == "Lumpsum":
                mf_amount = st.number_input("Amount (₹)", min_value=1_000,
                                            max_value=100_000_000, value=1_00_000,
                                            step=1_000, format="%d", key="mf_lumpsum")
                mf_stepup = 0.0
            else:
                mf_amount = st.number_input("Monthly SIP (₹)", min_value=500,
                                            max_value=10_000_000, value=5_000,
                                            step=500, format="%d", key="mf_sip_amt")
                mf_stepup = float(st.slider("Annual Step-up %", 0, 30, 0, key="mf_stepup"))
            st.divider()
            if "Historical" in mf_mode:
                st.markdown("### 📅 Period")
                mf_start = st.date_input(
                    "Invest from", key="mf_start_date",
                    value=date(TODAY.year - 5, TODAY.month, 1),
                    min_value=date(1995, 1, 1),
                    max_value=TODAY - timedelta(days=30),
                )
                if mf_inv_type == "SIP (Monthly)":
                    mf_end = st.date_input("SIP end date", key="mf_end_date",
                                           value=TODAY,
                                           min_value=mf_start + timedelta(days=30),
                                           max_value=TODAY)
                else:
                    mf_end = TODAY
            else:
                st.markdown("### 🔮 Projection")
                mf_proj_years = st.slider("Horizon (years)", 1, 40, 10, key="mf_years")
                mf_inflation  = st.slider("Inflation %", 0, 15, 6, key="mf_inflation")

    # ── Main content area ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1A2E4A 0%,#1B5299 100%);
         padding:16px 24px;border-radius:12px;margin-bottom:14px;
         border-bottom:3px solid #C8912A;">
      <h2 style="color:white;margin:0;font-size:20px;letter-spacing:0.5px;">
        📊 MUTUAL FUNDS INVESTMENT DASHBOARD
      </h2>
      <p style="color:#C8912A;margin:3px 0 0;font-size:11px;letter-spacing:0.3px;">
        Powered by mfapi.in &nbsp;|&nbsp; Real NAV · Historical Backtesting · Future Projection
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not selected_code:
        st.markdown("""
        <div style="background:#EBF5FB;border-left:4px solid #1B5299;border-radius:6px;
             padding:12px 16px;font-size:11px;color:#1A2E4A;margin-bottom:12px;">
        <b>How to use:</b> Search for a mutual fund using the sidebar (by AMC or by name).
        Then choose <b>Historical Backtest</b> to simulate past returns with real NAV data,
        or <b>Future Projection</b> to estimate growth over a chosen horizon.
        Toggle between <b>Lumpsum</b> and <b>SIP</b> investment modes.
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
        st.markdown('<div class="section-header">Popular Funds — Search by name in sidebar</div>',
                    unsafe_allow_html=True)
        pcols = st.columns(4)
        for i, (fname, code) in enumerate(pop.items()):
            with pcols[i % 4]:
                st.markdown(
                    f'<div class="metric-card" style="border-left-color:{MFC["blue"]}">'
                    f'<div class="metric-label">Code: {code}</div>'
                    f'<div class="metric-value" style="font-size:12px;color:{MFC["blue"]}">{fname}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        return

    # ── Fetch NAV ───────────────────────────────────────────────────────────────
    with st.spinner(f"Fetching NAV history…"):
        try:
            mf_meta, mf_navs = _mf_fetch_nav(selected_code)
        except Exception as e:
            st.error(f"Failed to fetch NAV: {e}")
            return

    if mf_navs.empty:
        st.error("No NAV data available for this fund.")
        return

    hist_ret   = _mf_hist_returns(mf_navs)
    latest_nav = mf_navs["nav"].iloc[-1]
    nav_start  = mf_navs["date"].iloc[0].date()
    nav_end    = mf_navs["date"].iloc[-1].date()
    nav_years  = (mf_navs["date"].iloc[-1] - mf_navs["date"].iloc[0]).days / 365.25

    # ── Fund header strip ──────────────────────────────────────────────────────
    fh1, fh2, fh3, fh4 = st.columns([4, 1.2, 1.2, 1.2])
    with fh1:
        st.markdown(f"### {selected_name}")
        badges = []
        for key, bg, fg in [
            ("scheme_category", "#EBF5FB", "#1B5299"),
            ("scheme_type",     "#EAFAF1", "#27AE60"),
            ("fund_house",      "#FEF9E7", "#C8912A"),
        ]:
            v = mf_meta.get(key, "")
            if v:
                badges.append(
                    f'<span style="display:inline-block;padding:3px 9px;border-radius:12px;'
                    f'font-size:10px;font-weight:700;background:{bg};color:{fg};margin:2px;">'
                    f'{v}</span>'
                )
        st.markdown(" ".join(badges), unsafe_allow_html=True)
        st.caption(
            f"Code: **{selected_code}** · Data: {nav_start.strftime('%d %b %Y')} → "
            f"{nav_end.strftime('%d %b %Y')} ({nav_years:.1f} yrs, {len(mf_navs):,} records)"
        )
    with fh2:
        nav_prev  = mf_navs["nav"].iloc[-2] if len(mf_navs) > 1 else latest_nav
        nav_delta = latest_nav - nav_prev
        st.metric("Latest NAV", f"₹{latest_nav:.4f}",
                  delta=f"{nav_delta:+.4f} ({nav_delta/nav_prev*100:+.2f}%)")
    with fh3:
        if "1yr" in hist_ret:
            st.metric("1yr Return", f"{hist_ret['1yr']:.2f}%")
    with fh4:
        if "3yr" in hist_ret:
            st.metric("3yr CAGR", f"{hist_ret['3yr']:.2f}%")

    # Historical CAGR strip
    if hist_ret:
        st.markdown('<div class="section-header">Historical CAGR Returns</div>',
                    unsafe_allow_html=True)
        rc = st.columns(len(hist_ret))
        for col, (period, ret) in zip(rc, hist_ret.items()):
            clr = MFC["green"] if ret >= 0 else MFC["red"]
            with col:
                st.markdown(
                    f'<div class="metric-card" style="border-left-color:{clr};text-align:center;">'
                    f'<div class="metric-label">{period} CAGR</div>'
                    f'<div class="metric-value" style="color:{clr}">{ret:.2f}%</div>'
                    f'<div class="metric-sub">annualised</div></div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ══ HISTORICAL BACKTEST ════════════════════════════════════════════════════
    if "Historical" in mf_mode:
        st.markdown('<div class="section-header">📅 Historical Backtest — Real NAV Data</div>',
                    unsafe_allow_html=True)

        if mf_inv_type == "Lumpsum":
            res = _mf_lumpsum(mf_navs, mf_start, mf_amount)
        else:
            res = _mf_sip(mf_navs, mf_start, mf_amount, mf_end, mf_stepup)

        if res is None:
            st.error("No NAV data from the selected start date. Try a later date.")
            return

        invested_ = res["invested"]
        end_val_  = res["end_value"]
        abs_ret_  = res["abs_return"]
        abs_pct_  = res["abs_pct"]
        years_    = res["years"]
        s_date_   = res["start_date"]
        e_date_   = res.get("end_date", nav_end)
        cagr_val_ = res["cagr"] if mf_inv_type == "Lumpsum" else res["cagr_approx"]
        gain_clr  = MFC["green"] if abs_ret_ >= 0 else MFC["red"]

        sub_lbl = (f"₹{mf_amount:,} one-time · {s_date_} → {e_date_}"
                   if mf_inv_type == "Lumpsum"
                   else f"₹{mf_amount:,}/mo × {res['installments']} instalments · {s_date_} → {e_date_}")

        # KPI cards
        kp = st.columns(5)
        kpis_ = [
            ("Invested",        fmt_inr(invested_),    sub_lbl[:40],                    MFC["blue"]),
            ("Current Value",   fmt_inr(end_val_),     f"As of {e_date_.strftime('%d %b %Y')}", MFC["teal"]),
            ("Absolute Return", fmt_inr(abs(abs_ret_)), f"{abs_pct_:+.2f}%",            gain_clr),
            ("CAGR (Est.)",     f"{cagr_val_:.2f}%",   "Compound annual growth rate",   MFC["purple"]),
            ("Duration",        f"{years_:.1f} yrs",   f"{int(years_*12)} months",      MFC["gold"]),
        ]
        for col, (lbl, val, sub, clr) in zip(kp, kpis_):
            with col:
                st.markdown(_mf_card(lbl, val, sub, clr), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────────────────
        if mf_inv_type == "Lumpsum":
            pf   = res["period_df"]
            fig1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig1.add_trace(go.Scatter(
                x=mf_navs["date"], y=mf_navs["nav"], name="NAV (full)",
                line=dict(color=COLORS["gray"], width=1), opacity=0.3,
            ), secondary_y=False)
            fig1.add_trace(go.Scatter(
                x=pf["date"], y=pf["nav"], name="NAV (period)",
                line=dict(color=MFC["blue"], width=2),
            ), secondary_y=False)
            fig1.add_trace(go.Scatter(
                x=pf["date"], y=pf["portfolio_value"], name="Portfolio Value",
                line=dict(color=MFC["green"], width=2.5),
                fill="tozeroy", fillcolor="rgba(39,174,96,0.08)",
            ), secondary_y=True)
            fig1.add_trace(go.Scatter(
                x=pf["date"], y=pf["invested"], name="Invested",
                line=dict(color=MFC["orange"], width=1.5, dash="dash"),
            ), secondary_y=True)
            fig1.add_vline(x=str(s_date_), line_dash="dot", line_color=MFC["gold"])
            fig1.add_annotation(x=str(s_date_), text=f"Buy {fmt_inr(mf_amount)}",
                                font=dict(size=10, color=MFC["gold"]),
                                showarrow=False, xanchor="left", yref="paper", y=0.98)
            fig1.update_layout(
                title=dict(text=f"NAV & Portfolio Growth · {selected_name[:55]}", font_size=13),
                height=380, hovermode="x unified",
                legend=dict(orientation="h", y=1.08, x=0, font_size=10),
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=60, b=40, l=50, r=50),
            )
            fig1.update_yaxes(title_text="NAV (₹)", secondary_y=False, showgrid=True, gridcolor="#ECF0F1")
            fig1.update_yaxes(title_text="Portfolio Value (₹)", secondary_y=True)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            fp   = res["full_period"]
            sipd = res["sip_df"]
            fig1 = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                                 subplot_titles=("Portfolio Value vs Invested", "Units Accumulated"))
            fig1.add_trace(go.Scatter(x=fp["date"], y=fp["portfolio_value"],
                                      name="Portfolio Value",
                                      line=dict(color=MFC["green"], width=2.5),
                                      fill="tozeroy", fillcolor="rgba(39,174,96,0.10)"), row=1, col=1)
            fig1.add_trace(go.Scatter(x=fp["date"], y=fp["cum_invested"],
                                      name="Amount Invested",
                                      line=dict(color=MFC["orange"], width=1.5, dash="dash")), row=1, col=1)
            fig1.add_trace(go.Scatter(x=sipd["date"], y=sipd["cum_units"],
                                      name="Cumulative Units",
                                      line=dict(color=MFC["purple"], width=2),
                                      fill="tozeroy", fillcolor="rgba(142,68,173,0.10)"), row=1, col=2)
            fig1.update_layout(height=380, hovermode="x unified",
                               legend=dict(orientation="h", y=1.1, x=0, font_size=10),
                               plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(t=60, b=40, l=50, r=50))
            st.plotly_chart(fig1, use_container_width=True)

        # ── Donut + Monthly returns ────────────────────────────────────────────
        dc1, dc2 = st.columns([1, 2])
        with dc1:
            st.markdown('<div class="section-header">Return Breakdown</div>',
                        unsafe_allow_html=True)
            donut = go.Figure(go.Pie(
                labels=["Invested", "Gains" if abs_ret_ >= 0 else "Loss"],
                values=[invested_, max(abs_ret_, 0)],
                hole=0.55,
                marker_colors=[MFC["blue"], MFC["green"] if abs_ret_ >= 0 else MFC["red"]],
                textinfo="label+percent", textfont_size=11,
            ))
            donut.update_layout(
                height=280, margin=dict(t=20, b=20, l=10, r=10),
                showlegend=False, paper_bgcolor="white",
                annotations=[dict(text=f"<b>{fmt_inr(end_val_)}</b><br>Total",
                                  x=0.5, y=0.5, showarrow=False, font_size=11)],
            )
            st.plotly_chart(donut, use_container_width=True)

        with dc2:
            st.markdown('<div class="section-header">Monthly NAV Returns (Period)</div>',
                        unsafe_allow_html=True)
            src = res["period_df"] if mf_inv_type == "Lumpsum" else res["full_period"]
            nav_monthly = src.set_index("date").resample("M")["nav"].last()
            nav_m_ret   = nav_monthly.pct_change().dropna() * 100
            bar_fig = go.Figure(go.Bar(
                x=nav_m_ret.index, y=nav_m_ret.values,
                marker_color=[MFC["green"] if v >= 0 else MFC["red"] for v in nav_m_ret],
            ))
            bar_fig.add_hline(y=0, line_color=COLORS["gray"], line_width=0.8)
            bar_fig.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
                                  margin=dict(t=10, b=30, l=40, r=20),
                                  yaxis=dict(title="Return %", gridcolor="#ECF0F1"),
                                  xaxis=dict(showgrid=False))
            st.plotly_chart(bar_fig, use_container_width=True)

        # ── NAV statistics row ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">NAV Statistics (Investment Period)</div>',
                    unsafe_allow_html=True)
        pnav = res["period_df"]["nav"] if mf_inv_type == "Lumpsum" else res["full_period"]["nav"]
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        nav_stats = [
            ("NAV at Entry",  f"₹{res['start_nav']:.4f}",  MFC["blue"]),
            ("NAV Today",     f"₹{res['end_nav']:.4f}",    MFC["teal"]),
            ("Period High",   f"₹{pnav.max():.4f}",        MFC["green"]),
            ("Period Low",    f"₹{pnav.min():.4f}",        MFC["red"]),
            ("Std Dev (NAV)", f"₹{pnav.std():.4f}",        MFC["purple"]),
            ("Monthly Vol",   f"{nav_m_ret.std():.2f}%",   MFC["orange"]),
        ]
        for col, (lbl, val, clr) in zip([s1, s2, s3, s4, s5, s6], nav_stats):
            with col:
                st.markdown(
                    f'<div class="metric-card" style="border-left-color:{clr}">'
                    f'<div class="metric-label">{lbl}</div>'
                    f'<div class="metric-value" style="color:{clr};font-size:15px">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if mf_inv_type == "SIP (Monthly)":
            with st.expander("📋 SIP Instalment Details", expanded=False):
                sd = res["sip_df"].copy()
                sd["date"]          = sd["date"].dt.strftime("%d %b %Y")
                sd["nav"]           = sd["nav"].apply(lambda x: f"₹{x:.4f}")
                sd["units_bought"]  = sd["units_bought"].apply(lambda x: f"{x:.4f}")
                sd["cum_units"]     = sd["cum_units"].apply(lambda x: f"{x:.4f}")
                sd["sip_amount"]    = sd["sip_amount"].apply(fmt_inr)
                sd["cum_invested"]  = sd["cum_invested"].apply(fmt_inr)
                sd.columns = ["Date", "NAV", "Units Bought", "Cum. Units",
                              "SIP Amount", "Cum. Invested"]
                st.dataframe(sd, use_container_width=True, hide_index=True)

    # ══ FUTURE PROJECTION ══════════════════════════════════════════════════════
    else:
        st.markdown('<div class="section-header">🔮 Future Projection — Return Simulator</div>',
                    unsafe_allow_html=True)

        # Return rate picker (shown in sidebar continuation)
        with st.sidebar:
            st.markdown("### 📈 Return Rate")
            rate_opts = {f"{k} CAGR ({v:.2f}%)": v for k, v in hist_ret.items()}
            rate_opts["Custom %"] = None
            rate_lbl = st.selectbox("Use rate from", list(rate_opts.keys()), key="mf_rate_lbl")
            if rate_opts[rate_lbl] is None:
                base_rate = float(st.number_input("Custom %", min_value=0.1, max_value=50.0,
                                                  value=12.0, step=0.5, key="mf_custom_rate"))
            else:
                base_rate = rate_opts[rate_lbl]
            st.caption(f"Base rate: **{base_rate:.2f}% p.a.**")

        bear_rate = max(base_rate * 0.55, 1.0)
        bull_rate = base_rate * 1.45
        inv_lbl   = "Lumpsum" if mf_inv_type == "Lumpsum" else "SIP (Monthly)"

        bear_df = _mf_project_future(mf_amount, inv_lbl, bear_rate, mf_proj_years, mf_stepup)
        base_df = _mf_project_future(mf_amount, inv_lbl, base_rate, mf_proj_years, mf_stepup)
        bull_df = _mf_project_future(mf_amount, inv_lbl, bull_rate, mf_proj_years, mf_stepup)

        invested_f = base_df["invested"].iloc[-1]
        end_bear   = bear_df["value"].iloc[-1]
        end_base   = base_df["value"].iloc[-1]
        end_bull   = bull_df["value"].iloc[-1]
        infl_adj   = end_base / ((1 + mf_inflation / 100) ** mf_proj_years)

        # KPI cards
        kp2 = st.columns(5)
        kpis2 = [
            ("Total Invested",   fmt_inr(invested_f),
             f"{mf_proj_years} yrs · {inv_lbl}", MFC["blue"]),
            ("Bear Case",        fmt_inr(end_bear),
             f"{bear_rate:.1f}% p.a. · gain: {fmt_inr(end_bear - invested_f)}", MFC["orange"]),
            ("Base Case",        fmt_inr(end_base),
             f"{base_rate:.1f}% p.a. · gain: {fmt_inr(end_base - invested_f)}", MFC["green"]),
            ("Bull Case",        fmt_inr(end_bull),
             f"{bull_rate:.1f}% p.a. · gain: {fmt_inr(end_bull - invested_f)}", MFC["purple"]),
            ("Inflation-Adj.",   fmt_inr(infl_adj),
             f"@ {mf_inflation}% inflation over {mf_proj_years}y", MFC["gold"]),
        ]
        for col, (lbl, val, sub, clr) in zip(kp2, kpis2):
            with col:
                st.markdown(_mf_card(lbl, val, sub, clr), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Growth chart
        x_vals = base_df["year"]
        fig_p  = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=list(x_vals) + list(x_vals[::-1]),
            y=list(bull_df["value"]) + list(bear_df["value"][::-1]),
            fill="toself", fillcolor="rgba(39,174,96,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Bear–Bull Range", hoverinfo="skip",
        ))
        fig_p.add_trace(go.Scatter(x=x_vals, y=bear_df["value"],
                                   name=f"Bear ({bear_rate:.1f}%)",
                                   line=dict(color=MFC["orange"], width=1.5, dash="dot")))
        fig_p.add_trace(go.Scatter(x=x_vals, y=base_df["value"],
                                   name=f"Base ({base_rate:.1f}%)",
                                   line=dict(color=MFC["green"], width=2.5)))
        fig_p.add_trace(go.Scatter(x=x_vals, y=bull_df["value"],
                                   name=f"Bull ({bull_rate:.1f}%)",
                                   line=dict(color=MFC["purple"], width=1.5, dash="dash")))
        fig_p.add_trace(go.Scatter(x=x_vals, y=base_df["invested"],
                                   name="Invested Capital",
                                   line=dict(color=MFC["blue"], width=2, dash="dashdot")))
        infl_line = [end_base / ((1 + mf_inflation / 100) ** (mf_proj_years - yr)) for yr in x_vals]
        fig_p.add_trace(go.Scatter(x=x_vals, y=infl_line,
                                   name=f"Infl-adj ({mf_inflation}%)",
                                   line=dict(color=MFC["gray"], width=1, dash="dot"), opacity=0.6))
        fig_p.update_layout(
            title=dict(text=f"Projected Growth · {mf_proj_years}yr · {inv_lbl} · {selected_name[:50]}",
                       font_size=13),
            height=420, hovermode="x unified",
            xaxis=dict(title="Years", showgrid=False),
            yaxis=dict(title="Value (₹)", showgrid=True, gridcolor="#ECF0F1"),
            legend=dict(orientation="h", y=1.08, x=0, font_size=10),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=60, b=40, l=60, r=30),
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # Donut + Milestone table
        mc1, mc2 = st.columns([1, 2])
        with mc1:
            st.markdown('<div class="section-header">Invested vs Gains (Base)</div>',
                        unsafe_allow_html=True)
            base_gains = end_base - invested_f
            d2 = go.Figure(go.Pie(
                labels=["Invested", "Estimated Gains"],
                values=[invested_f, max(base_gains, 0)],
                hole=0.55,
                marker_colors=[MFC["blue"], MFC["green"]],
                textinfo="label+percent", textfont_size=11,
            ))
            d2.update_layout(
                height=300, margin=dict(t=20, b=20, l=10, r=10),
                showlegend=False, paper_bgcolor="white",
                annotations=[dict(text=f"<b>{fmt_inr(end_base)}</b><br>@ {base_rate:.1f}%",
                                  x=0.5, y=0.5, showarrow=False, font_size=11)],
            )
            st.plotly_chart(d2, use_container_width=True)
        with mc2:
            st.markdown('<div class="section-header">Year-by-Year Milestones (Base Case)</div>',
                        unsafe_allow_html=True)
            milestones = []
            for yr in range(1, mf_proj_years + 1):
                rb = bear_df[bear_df["month"] == yr * 12].iloc[0]
                rm = base_df[base_df["month"] == yr * 12].iloc[0]
                ru = bull_df[bull_df["month"] == yr * 12].iloc[0]
                milestones.append({
                    "Year":                     yr,
                    "Invested":                 fmt_inr(rm["invested"]),
                    f"Bear ({bear_rate:.1f}%)": fmt_inr(rb["value"]),
                    f"Base ({base_rate:.1f}%)": fmt_inr(rm["value"]),
                    f"Bull ({bull_rate:.1f}%)": fmt_inr(ru["value"]),
                    "Gain % (Base)":            f"{rm['gain_pct']:.1f}%" if pd.notna(rm["gain_pct"]) else "—",
                })
            st.dataframe(pd.DataFrame(milestones), use_container_width=True, hide_index=True)

        # Scenario comparison using all historical CAGRs
        if hist_ret:
            st.markdown('<div class="section-header">Scenario Comparison — All Historical CAGRs</div>',
                        unsafe_allow_html=True)
            cmp_rows = []
            for period, rate in hist_ret.items():
                df_tmp  = _mf_project_future(mf_amount, inv_lbl, rate, mf_proj_years, mf_stepup)
                end_tmp = df_tmp["value"].iloc[-1]
                inv_tmp = df_tmp["invested"].iloc[-1]
                cmp_rows.append({
                    "Period Used":     period,
                    "CAGR":            f"{rate:.2f}%",
                    "Invested":        fmt_inr(inv_tmp),
                    "Projected Value": fmt_inr(end_tmp),
                    "Total Gain":      fmt_inr(end_tmp - inv_tmp),
                    "Return %":        f"{(end_tmp - inv_tmp) / inv_tmp * 100:.1f}%",
                    "Infl-adj":        fmt_inr(end_tmp / ((1 + mf_inflation / 100) ** mf_proj_years)),
                })
            st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

        st.markdown(
            f'<div style="background:#EBF5FB;border-left:4px solid #1B5299;border-radius:6px;'
            f'padding:10px 14px;font-size:11px;color:#1A2E4A;margin-top:10px;">'
            f'<b>Disclaimer:</b> Projections are estimates based on historical CAGR from live NAV data. '
            f'Mutual fund returns are subject to market risk. Past performance is not indicative of future results. '
            f'Bear/Bull at {bear_rate:.1f}% / {bull_rate:.1f}% (55%/145% of base).</div>',
            unsafe_allow_html=True,
        )

    # ── Full NAV history chart ─────────────────────────────────────────────────
    with st.expander("📈 Full NAV History Chart", expanded=False):
        fig_full = go.Figure()
        fig_full.add_trace(go.Scatter(
            x=mf_navs["date"], y=mf_navs["nav"],
            line=dict(color=MFC["blue"], width=1.5),
            fill="tozeroy", fillcolor="rgba(27,82,153,0.07)",
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
        tail = mf_navs.tail(90).copy()
        tail["date"] = tail["date"].dt.strftime("%d %b %Y")
        tail["nav"]  = tail["nav"].apply(lambda x: f"₹{x:.4f}")
        tail.columns = ["Date", "NAV"]
        st.dataframe(tail[::-1], use_container_width=True, hide_index=True)

    st.caption(
        f"Data: [mfapi.in](https://www.mfapi.in) — free open NAV API · "
        f"Cache: 5 min · {TODAY.strftime('%d %b %Y')}"
    )

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── App header + page navigation ──────────────────────────────────────────
    st.markdown("""
    <div style="background:#1A2E4A;padding:14px 16px;border-radius:8px;margin-bottom:10px;">
      <div style="color:#C8912A;font-size:10px;font-weight:700;letter-spacing:1px;">
        PERSONAL FINANCE
      </div>
      <div style="color:white;font-size:13px;font-weight:600;margin-top:4px;">
        Analytics Dashboard
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.selectbox(
        "Navigate to",
        ["🏦 Loan Dashboard", "📊 Mutual Funds"],
        key="current_page",
        label_visibility="collapsed",
    )

    st.caption(f"**{TODAY.strftime('%d %b %Y')}**  ·  "
               f"User: **{st.session_state.get('username', '')}**")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    st.divider()

    if _on_loan_page:
        st.markdown(f"""
        <div style="background:#1B5299;padding:10px 14px;border-radius:8px;margin-bottom:10px;">
          <div style="color:#C8912A;font-size:10px;font-weight:700;letter-spacing:1px;">
            EDUCATION LOAN
          </div>
          <div style="color:white;font-size:12px;font-weight:600;margin-top:3px;">{BANK}</div>
          <div style="color:#BDC3C7;font-size:11px;">Acct: {MASKED_ACCT}
          &nbsp;·&nbsp; {RATE_PA*100:.2f}% p.a.</div>
        </div>
        """, unsafe_allow_html=True)

    if _on_loan_page:
        # ── Monthly EMI slider ────────────────────────────────────────────────────
        st.markdown("### ⚙️ Monthly EMI")
        monthly_pmt = st.slider(
            "Monthly Payment (₹)",
            min_value=10_000, max_value=2_00_000,
            value=st.session_state.default_emi,
            step=5_000, format="₹%d",
            key="emi_slider",
        )
        st.session_state.default_emi = monthly_pmt
        delta_color = "normal" if monthly_pmt > breakeven else "inverse"
        st.metric(
            "Monthly Breakeven",
            fmt_inr(breakeven),
            delta=f"{'surplus' if monthly_pmt > breakeven else 'deficit'}: "
                  f"{fmt_inr(abs(monthly_pmt - breakeven))}",
            delta_color=delta_color,
        )

        st.divider()

        # ── Add Manual Payment ────────────────────────────────────────────────────
        st.markdown("### ➕ Add / Update Payment")
        with st.form("add_payment_form", clear_on_submit=True):
            pay_date   = st.date_input("Payment Date", value=TODAY,
                                       min_value=date(2020, 1, 1), max_value=TODAY)
            pay_amount = st.number_input("Amount (₹)", min_value=500,
                                         max_value=50_00_000, value=20_000, step=500)
            pay_remark = st.text_input("Remarks", placeholder="e.g. Extra EMI, NEFT")
            if st.form_submit_button("💾 Save Payment", use_container_width=True):
                save_manual_payment(pay_date, pay_amount,
                                    pay_remark or "Manual Payment")
                st.session_state.manual_version += 1
                st.cache_data.clear()
                st.success(f"Saved {fmt_inr(pay_amount)} on {pay_date}")
                st.rerun()

        manual_list = load_manual_payments()
        if manual_list:
            st.markdown(f"**{len(manual_list)} manual entry(s):**")
            for i, p in enumerate(manual_list):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.caption(f"{p['date']}: **{fmt_inr(p['amount'])}**  \n"
                               f"_{p.get('remarks','')[:28]}_")
                with c2:
                    if st.button("✕", key=f"del_{i}", help="Delete this entry"):
                        delete_manual_payment(i)
                        st.session_state.manual_version += 1
                        st.cache_data.clear()
                        st.rerun()
        else:
            st.caption("No manual payments yet.")

        st.divider()

        # ── Date Range Filter ─────────────────────────────────────────────────────
        st.markdown("### 📅 Date Range")
        min_date   = disb["Date"].min().date()
        date_range = st.date_input(
            "Filter charts from",
            value=(min_date, TODAY),
            min_value=min_date, max_value=TODAY,
        )
        filter_start = pd.Timestamp(date_range[0]) if len(date_range) >= 1 else pd.Timestamp(min_date)
        filter_end   = pd.Timestamp(date_range[1]) if len(date_range) == 2 else pd.Timestamp(TODAY)

        st.divider()
        auto_refresh = st.toggle("Auto-refresh (5 min)", value=False)
        if auto_refresh:
            st.caption("Page will refresh every 5 minutes")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE ROUTER — render MF page and stop; loan content falls through below
# ─────────────────────────────────────────────────────────────────────────────
if not _on_loan_page:
    render_mf_page()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER  (Loan Dashboard)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#1A2E4A 0%,#1B5299 100%);
     padding:16px 24px;border-radius:12px;margin-bottom:14px;
     border-bottom:3px solid #C8912A;">
  <h2 style="color:white;margin:0;font-size:19px;letter-spacing:0.5px;">
    📈 EDUCATION LOAN ANALYTICS DASHBOARD
  </h2>
  <p style="color:#C8912A;margin:3px 0 0;font-size:11px;letter-spacing:0.3px;">
    {BANK} &nbsp;|&nbsp; Account: {MASKED_ACCT}
    &nbsp;|&nbsp; Rate: {RATE_PA*100:.2f}% p.a.
    &nbsp;|&nbsp; As on: {TODAY.strftime('%d %b %Y')}
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  KPI CARDS  —  2 rows × 4
# ─────────────────────────────────────────────────────────────────────────────
row1 = [
    ("Total Disbursed",  fmt_inr(total_disbursed),
     f"{len(disb)} tranches · {disb_meta[0]['tag']} – {disb_meta[-1]['tag']}",
     COLORS["blue"]),
    ("Outstanding",      fmt_inr(outstanding),
     f"Balance as on {TODAY.strftime('%d %b %Y')}",
     COLORS["red"]),
    ("Total Repaid",     fmt_inr(total_repaid),
     "All payments made (principal + interest collections)",
     COLORS["green"]),
    ("Principal Repaid", fmt_inr(principal_repaid),
     f"{principal_repaid/total_disbursed*100:.1f}% of total disbursed",
     COLORS["teal"]),
]
row2 = [
    ("Interest Charged", fmt_inr(total_int_charged),
     f"Bank charges · {n_ints} monthly entries",
     COLORS["orange"]),
    ("Theory Int @8.55%", fmt_inr(theory_total_int),
     "Daily accrual on actual outstanding",
     COLORS["gold"]),
    ("Loan Age",         f"{int(loan_age_months)}m",
     f"{loan_age_days} days · since {loan_start.strftime('%b %Y')}",
     COLORS["purple"]),
    ("Monthly Breakeven", fmt_inr(breakeven),
     "Min payment to stop principal growing",
     COLORS["gray"]),
]
for row in [row1, row2]:
    cols = st.columns(4)
    for col, (lbl, val, sub, clr) in zip(cols, row):
        with col:
            st.markdown(
                f'<div class="metric-card" style="border-left-color:{clr}">'
                f'<div class="metric-label">{lbl}</div>'
                f'<div class="metric-value" style="color:{clr}">{val}</div>'
                f'<div class="metric-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🔍 Deep Analysis",
    "📅 Repayment Planner",
    "💰 Prepayment Calculator",
    "📋 Transaction Log",
    "ℹ️ Key Metrics",
])

bal_filtered = bal_ts[
    (bal_ts["Date"] >= filter_start) & (bal_ts["Date"] <= filter_end)
]

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Outstanding Loan Balance Over Time</div>',
                unsafe_allow_html=True)

    fig_bal = go.Figure()
    fig_bal.add_trace(go.Scatter(
        x=bal_filtered["Date"], y=bal_filtered["Outs"],
        fill="tozeroy", fillcolor="rgba(27,82,153,0.10)",
        line=dict(color=COLORS["blue"], width=2.5),
        name="Outstanding",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Outstanding: ₹%{y:,.0f}<extra></extra>",
    ))
    for i, row in disb.iterrows():
        if not (filter_start <= row["Date"] <= filter_end):
            continue
        bal_at = bal_ts[bal_ts["Date"] <= row["Date"]]["Outs"]
        y_val  = float(bal_at.iloc[-1]) if len(bal_at) else 0
        col    = TRANCHE_COLS[i % 4]
        fig_bal.add_vline(x=row["Date"], line_dash="dash",
                          line_color=col, line_width=1.5, opacity=0.6)
        fig_bal.add_annotation(
            x=row["Date"], y=y_val,
            text=f"<b>D{i+1}</b><br>{fmt_inr(row['Abs'])}",
            showarrow=True, arrowhead=2, arrowcolor=col,
            ax=0, ay=-45, font=dict(size=10, color=col),
            bgcolor="white", bordercolor=col, borderwidth=1,
        )
    fig_bal.update_layout(
        height=340, margin=dict(t=10, b=40, l=70, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE",
                   title="Outstanding (₹)", tickprefix="₹", tickformat=",.0f"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_bal, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Disbursements (Tranches)</div>',
                    unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            values=[d["amt"] for d in disb_meta],
            labels=[f"T{d['n']} ({d['tag']}): {fmt_inr(d['amt'])}" for d in disb_meta],
            marker=dict(colors=TRANCHE_COLS[:len(disb_meta)],
                        line=dict(color="white", width=2)),
            hole=0.55,
            texttemplate="%{percent:.1%}",
            hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<extra></extra>",
        ))
        fig_pie.add_annotation(
            text=f"<b>{fmt_inr(total_disbursed)}</b><br><span style='font-size:10px'>Total</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=COLORS["navy"]),
        )
        fig_pie.update_layout(
            height=320, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="white",
            legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=10)),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Cumulative: Interest Charged vs Total Repaid</div>',
                    unsafe_allow_html=True)
        ints_f   = ints_s[(ints_s["Date"] >= filter_start) & (ints_s["Date"] <= filter_end)]
        pmts_f   = pmts_s[(pmts_s["Date"] >= filter_start) & (pmts_s["Date"] <= filter_end)]
        theory_f = theory[(theory["Date"] >= filter_start) & (theory["Date"] <= filter_end)]

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=theory_f["Date"], y=theory_f["CumInt"] / 1e5,
            line=dict(color=COLORS["red"], width=1.6, dash="dash"),
            name=f"Theory @{RATE_PA*100:.2f}% ({fmt_inr(theory_total_int)})",
            hovertemplate="Theory: ₹%{y:.2f}L<extra></extra>",
        ))
        fig_cum.add_trace(go.Scatter(
            x=ints_f["Date"], y=ints_f["CumInt"] / 1e5,
            fill="tozeroy", fillcolor="rgba(230,126,34,0.12)",
            line=dict(color=COLORS["orange"], width=2.2),
            name=f"Interest Charged ({fmt_inr(total_int_charged)})",
            hovertemplate="Interest: ₹%{y:.2f}L<extra></extra>",
        ))
        fig_cum.add_trace(go.Scatter(
            x=pmts_f["Date"], y=pmts_f["CumPmt"] / 1e5,
            fill="tozeroy", fillcolor="rgba(39,174,96,0.12)",
            line=dict(color=COLORS["green"], width=2.2),
            name=f"Total Repaid ({fmt_inr(total_repaid)})",
            hovertemplate="Repaid: ₹%{y:.2f}L<extra></extra>",
        ))
        fig_cum.update_layout(
            height=320, margin=dict(t=10, b=50, l=70, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#EEEEEE",
                       title="₹ Lakhs", ticksuffix="L"),
            legend=dict(orientation="h", yanchor="top", y=-0.18, font=dict(size=10)),
            hovermode="x unified",
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    st.markdown('<div class="section-header">Monthly Interest Charged (bars) + Cumulative (line)</div>',
                unsafe_allow_html=True)
    mon_int_f = mon_int[(mon_int["Dt"] >= filter_start) & (mon_int["Dt"] <= filter_end)]
    fig_mi    = make_subplots(specs=[[{"secondary_y": True}]])
    fig_mi.add_trace(go.Bar(
        x=mon_int_f["Dt"], y=mon_int_f["Abs"],
        name="Monthly Interest", marker_color=COLORS["orange"], opacity=0.8,
        hovertemplate="<b>%{x|%b %Y}</b><br>Interest: ₹%{y:,.0f}<extra></extra>",
    ), secondary_y=False)
    fig_mi.add_trace(go.Scatter(
        x=mon_int_f["Dt"], y=mon_int_f["Abs"].cumsum() / 1e5,
        name="Cumulative (₹L)", line=dict(color=COLORS["red"], width=2.5),
        mode="lines+markers", marker=dict(size=5),
        hovertemplate="<b>%{x|%b %Y}</b><br>Cumulative: ₹%{y:.2f}L<extra></extra>",
    ), secondary_y=True)
    fig_mi.update_layout(
        height=320, margin=dict(t=10, b=40, l=70, r=70),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.05), hovermode="x unified", bargap=0.2,
    )
    fig_mi.update_yaxes(title_text="Monthly Interest (₹)", secondary_y=False,
                        tickprefix="₹", showgrid=True, gridcolor="#EEEEEE")
    fig_mi.update_yaxes(title_text="Cumulative (₹ Lakhs)", secondary_y=True,
                        ticksuffix="L", showgrid=False)
    st.plotly_chart(fig_mi, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2 — DEEP ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Disbursement Timeline</div>',
                unsafe_allow_html=True)
    fig_tl = go.Figure()
    fig_tl.add_shape(type="line", x0=0, x1=1, y0=0, y1=0,
                     xref="paper", yref="y",
                     line=dict(color=COLORS["navy"], width=3))
    for d in disb_meta:
        fig_tl.add_trace(go.Scatter(
            x=[pd.Timestamp(d["dt"])], y=[0],
            mode="markers+text",
            marker=dict(size=30, color=d["col"], line=dict(color="white", width=3)),
            text=[str(d["n"])],
            textfont=dict(color="white", size=12, family="Arial Black"),
            textposition="middle center",
            name=f"Tranche {d['n']} ({d['tag']}): {fmt_inr(d['amt'])}",
            hovertemplate=(
                f"<b>Tranche {d['n']} — {d['tag']}</b><br>"
                f"Date: {d['dt'].strftime('%d %b %Y')}<br>"
                f"Amount: {fmt_inr_full(d['amt'])}<br>"
                f"Days elapsed: {d['days']}<br>"
                f"Theory int: {fmt_inr(d['theory_int'])}<extra></extra>"
            ),
        ))
        label_y = 0.55 if d["n"] % 2 == 1 else -0.55
        fig_tl.add_annotation(
            x=pd.Timestamp(d["dt"]), y=label_y,
            text=f"<b>T{d['n']}</b><br>{d['tag']}<br>{fmt_inr(d['amt'])}",
            showarrow=True, arrowhead=0, ax=0,
            ay=40 if label_y < 0 else -40,
            font=dict(size=11, color=d["col"]), align="center",
        )
    fig_tl.add_vline(x=str(TODAY), line_dash="dash",
                     line_color=COLORS["gold"], line_width=2)
    fig_tl.add_annotation(
        x=str(TODAY), y=0.9, yref="paper",
        text=f"<b>Today ({TODAY.strftime('%d %b %Y')})</b>",
        showarrow=False, font=dict(color=COLORS["gold"], size=11),
        bgcolor="white", bordercolor=COLORS["gold"], borderwidth=1,
    )
    fig_tl.update_layout(
        height=280, margin=dict(t=20, b=20, l=20, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickformat="%b %Y"),
        yaxis=dict(visible=False, range=[-1.2, 1.2]),
        legend=dict(orientation="h", y=-0.12, font=dict(size=11)),
        hovermode="closest",
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Per-Tranche: Principal vs Theory Interest</div>',
                    unsafe_allow_html=True)
        fig_da  = go.Figure()
        xlabels = [f"T{d['n']} ({d['tag']})<br>{d['days']}d" for d in disb_meta]
        fig_da.add_trace(go.Bar(
            x=xlabels, y=[d["amt"] / 1e5 for d in disb_meta],
            name="Principal", marker=dict(color=TRANCHE_COLS[:len(disb_meta)], opacity=0.75),
            text=[f"₹{d['amt']/1e5:.1f}L" for d in disb_meta], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Principal: ₹%{y:.2f}L<extra></extra>",
        ))
        fig_da.add_trace(go.Bar(
            x=xlabels, y=[d["theory_int"] / 1e5 for d in disb_meta],
            name="Theory Interest",
            marker=dict(color=TRANCHE_COLS[:len(disb_meta)], opacity=1.0,
                        pattern=dict(shape="/", fgcolor="white")),
            text=[f"₹{d['theory_int']/1e5:.1f}L" for d in disb_meta], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Theory int: ₹%{y:.2f}L<extra></extra>",
        ))
        fig_da.update_layout(
            height=360, barmode="group",
            margin=dict(t=10, b=60, l=60, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#EEEEEE",
                       title="₹ Lakhs", ticksuffix="L"),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_da, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Effective Interest Rate Per Period</div>',
                    unsafe_allow_html=True)
        st.caption("Back-calculated from bank charges vs. outstanding balance")
        fig_ir = go.Figure()
        if len(rate_df):
            rate_f = rate_df[(rate_df["Date"] >= filter_start) & (rate_df["Date"] <= filter_end)]
            fig_ir.add_trace(go.Scatter(
                x=rate_f["Date"], y=rate_f["Rate"],
                mode="lines+markers", line=dict(color=COLORS["purple"], width=2),
                marker=dict(size=7, symbol="diamond"),
                name="Effective Rate",
                hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y:.2f}%<extra></extra>",
            ))
        fig_ir.add_hline(y=RATE_PA * 100, line_dash="dash",
                         line_color=COLORS["blue"], line_width=2,
                         annotation_text=f"{RATE_PA*100:.2f}% reference",
                         annotation_position="top right",
                         annotation_font=dict(color=COLORS["blue"]))
        fig_ir.update_layout(
            height=360, margin=dict(t=10, b=40, l=70, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#EEEEEE",
                       title="Effective Annual Rate (%)", ticksuffix="%"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_ir, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-header">Monthly: Interest Charged + Payments Made</div>',
                    unsafe_allow_html=True)
        all_months = pd.period_range(
            raw_df["Date"].min().to_period("M"),
            pd.Timestamp(TODAY).to_period("M"), freq="M")
        mi_map = mon_int.set_index("Month")["Abs"].to_dict()
        mp_map = mon_pmt.set_index("Month")["Abs"].to_dict()
        m_periods = [p for p in all_months
                     if filter_start <= pd.Timestamp(p.to_timestamp()) <= filter_end]
        m_dates = [p.to_timestamp() for p in m_periods]
        m_int_v = [mi_map.get(str(p), 0) for p in m_periods]
        m_pmt_v = [mp_map.get(str(p), 0) for p in m_periods]

        fig_mp = go.Figure()
        fig_mp.add_trace(go.Bar(
            x=m_dates, y=m_int_v, name="Interest Charged",
            marker_color=COLORS["orange"],
            hovertemplate="<b>%{x|%b %Y}</b><br>Interest: ₹%{y:,.0f}<extra></extra>",
        ))
        fig_mp.add_trace(go.Bar(
            x=m_dates, y=m_pmt_v, name="Payments Made",
            marker_color=COLORS["green"],
            hovertemplate="<b>%{x|%b %Y}</b><br>Paid: ₹%{y:,.0f}<extra></extra>",
        ))
        fig_mp.update_layout(
            barmode="group", height=340,
            margin=dict(t=10, b=40, l=70, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False, tickformat="%b'%y"),
            yaxis=dict(showgrid=True, gridcolor="#EEEEEE",
                       title="Amount (₹)", tickprefix="₹"),
            legend=dict(orientation="h", y=1.05),
            hovermode="x unified", bargap=0.15,
        )
        st.plotly_chart(fig_mp, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">Repayment Projection Scenarios</div>',
                    unsafe_allow_html=True)
        fig_proj = go.Figure()
        scenarios = [
            (20_000, COLORS["red"],    "dash"),
            (30_000, COLORS["orange"], "solid"),
            (50_000, COLORS["blue"],   "solid"),
            (75_000, COLORS["green"],  "solid"),
            (monthly_pmt, COLORS["gold"], "dot"),
        ]
        for pmt, col, dash in scenarios:
            proj   = project(pmt, outstanding)
            x_proj = [pd.Timestamp(TODAY) + pd.DateOffset(months=int(mo))
                      for mo in proj["month"]]
            label  = (f"₹{pmt:,}/mo — Slider ({months_str(proj)})"
                      if pmt == monthly_pmt else
                      f"₹{pmt:,}/mo ({months_str(proj)})")
            fig_proj.add_trace(go.Scatter(
                x=x_proj, y=proj["bal"] / 1e5, name=label,
                line=dict(color=col, width=2.5, dash=dash),
                hovertemplate=f"₹{pmt:,}/mo · %{{x|%b %Y}}: ₹%{{y:.2f}}L<extra></extra>",
            ))
        fig_proj.add_hline(y=0, line_color=COLORS["navy"], line_width=1, opacity=0.3)
        fig_proj.update_layout(
            height=340, margin=dict(t=10, b=50, l=70, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False, tickformat="%b'%y"),
            yaxis=dict(showgrid=True, gridcolor="#EEEEEE",
                       title="Outstanding (₹ Lakhs)", ticksuffix="L"),
            legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
            hovermode="x unified",
        )
        st.plotly_chart(fig_proj, use_container_width=True)

    st.markdown('<div class="section-header">Tranche Detail</div>', unsafe_allow_html=True)
    tc = st.columns(len(disb_meta))
    for col, d in zip(tc, disb_meta):
        with col:
            pct = d["theory_int"] / d["amt"] * 100 if d["amt"] else 0
            st.markdown(
                f'<div class="metric-card" style="border-left-color:{d["col"]}">'
                f'<div class="metric-label">Tranche {d["n"]} · {d["tag"]}</div>'
                f'<div class="metric-value" style="color:{d["col"]}">{fmt_inr(d["amt"])}</div>'
                f'<div class="metric-sub">'
                f'📅 {d["dt"].strftime("%d %b %Y")}<br>'
                f'⏱️ {d["days"]} days elapsed<br>'
                f'💸 Theory int: {fmt_inr(d["theory_int"])} ({pct:.1f}%)'
                f'</div></div>',
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 3 — REPAYMENT PLANNER
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Repayment Projection Scenarios</div>',
                unsafe_allow_html=True)

    ci1, ci2, ci3 = st.columns(3)
    with ci1: st.metric("Current Outstanding", fmt_inr_full(outstanding))
    with ci2: st.metric("Monthly Interest Accrual", fmt_inr(breakeven))
    with ci3:
        st.metric("Slider Payment", f"₹{monthly_pmt:,}/mo",
                  delta=f"{fmt_inr(monthly_pmt - breakeven)} to principal")

    fig_p2 = go.Figure()
    for pmt, col, dash in [
        (20_000, COLORS["red"],    "dash"),
        (30_000, COLORS["orange"], "solid"),
        (50_000, COLORS["blue"],   "solid"),
        (75_000, COLORS["green"],  "solid"),
        (1_00_000, COLORS["teal"], "solid"),
        (monthly_pmt, COLORS["gold"], "dot"),
    ]:
        if pmt == monthly_pmt and pmt in [20_000, 30_000, 50_000, 75_000, 1_00_000]:
            continue
        proj   = project(pmt, outstanding)
        x_proj = [pd.Timestamp(TODAY) + pd.DateOffset(months=int(mo))
                  for mo in proj["month"]]
        label  = (f"₹{pmt:,}/mo — Slider ({months_str(proj)})"
                  if pmt == monthly_pmt else f"₹{pmt:,}/mo ({months_str(proj)})")
        fig_p2.add_trace(go.Scatter(
            x=x_proj, y=proj["bal"] / 1e5, name=label,
            line=dict(color=col, width=2.5, dash=dash),
            hovertemplate=f"₹{pmt:,}/mo · %{{x|%b %Y}}: ₹%{{y:.2f}}L<extra></extra>",
        ))
    fig_p2.add_hline(y=0, line_color=COLORS["navy"], line_width=1, opacity=0.3)
    fig_p2.update_layout(
        height=400, margin=dict(t=10, b=70, l=70, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickformat="%b'%y"),
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE",
                   title="Outstanding (₹ Lakhs)", ticksuffix="L"),
        legend=dict(orientation="h", y=-0.2, font=dict(size=11)),
        hovermode="x unified",
    )
    st.plotly_chart(fig_p2, use_container_width=True)

    st.markdown('<div class="section-header">Interest vs Principal Split — Slider Scenario</div>',
                unsafe_allow_html=True)
    proj_custom        = project(monthly_pmt, outstanding)
    proj_custom["lbl"] = [
        (pd.Timestamp(TODAY) + pd.DateOffset(months=int(mo))).strftime("%b'%y")
        for mo in proj_custom["month"]
    ]
    fig_split = go.Figure()
    fig_split.add_trace(go.Bar(
        x=proj_custom["lbl"], y=proj_custom["principal"],
        name="Principal", marker_color=COLORS["blue"],
        hovertemplate="%{x}<br>Principal: ₹%{y:,.0f}<extra></extra>",
    ))
    fig_split.add_trace(go.Bar(
        x=proj_custom["lbl"], y=proj_custom["interest"],
        name="Interest", marker_color=COLORS["orange"],
        hovertemplate="%{x}<br>Interest: ₹%{y:,.0f}<extra></extra>",
    ))
    fig_split.update_layout(
        barmode="stack", height=320,
        margin=dict(t=10, b=60, l=70, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickangle=45,
                   title=f"Month (₹{monthly_pmt:,}/mo scenario)"),
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE",
                   title="Amount (₹)", tickprefix="₹"),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_split, use_container_width=True)

    st.markdown('<div class="section-header">Scenario Comparison</div>',
                unsafe_allow_html=True)
    rows = []
    for pmt in sorted(set([20_000, 30_000, 50_000, 75_000, 1_00_000, monthly_pmt])):
        p   = project(pmt, outstanding)
        n   = len(p) if p["bal"].iloc[-1] <= 100 else None
        ti  = p["interest"].sum()
        p50 = project(50_000, outstanding)["interest"].sum()
        rows.append({
            "Monthly Payment":  f"₹{pmt:,}",
            "Payoff Duration":  months_str(p),
            "Total Interest":   fmt_inr_full(ti) if n else "N/A",
            "Total Outflow":    fmt_inr_full(outstanding + ti) if n else "N/A",
            "vs ₹50K/mo":       ("—" if pmt == 50_000 else
                                 (f"saves {fmt_inr(p50 - ti)}" if n else "—")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 4 — PREPAYMENT CALCULATOR
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Lump-Sum Prepayment Impact Calculator</div>',
                unsafe_allow_html=True)
    st.caption(
        "Select a future date and lump-sum amount to see exactly how much interest "
        "you save and how many months you cut off the loan."
    )

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        prepay_date = st.date_input(
            "Prepayment Date",
            value=TODAY + timedelta(days=30),
            min_value=TODAY + timedelta(days=1),
            max_value=TODAY + timedelta(days=365 * 10),
            key="prepay_date",
        )
    with pcol2:
        lump_sum = st.number_input(
            "Lump Sum Amount (₹)",
            min_value=10_000, max_value=int(outstanding),
            value=min(1_00_000, int(outstanding)),
            step=10_000,
            key="lump_sum",
        )
    with pcol3:
        emi_for_calc = st.number_input(
            "Monthly EMI (₹)",
            min_value=10_000, max_value=2_00_000,
            value=monthly_pmt, step=5_000,
            key="emi_calc",
            help="EMI you plan to pay each month (use sidebar slider or set here)",
        )

    # Compute months from today to prepayment date
    days_to_prepay   = (prepay_date - TODAY).days
    months_to_prepay = max(1, round(days_to_prepay / 30.44))

    proj_without = project(emi_for_calc, outstanding)
    proj_with    = project_with_lumpsum(emi_for_calc, outstanding,
                                        lump_sum, months_to_prepay)

    dur_without  = len(proj_without) if proj_without["bal"].iloc[-1] <= 100 else None
    dur_with     = len(proj_with)    if proj_with["bal"].iloc[-1]    <= 100 else None
    int_without  = proj_without["interest"].sum()
    int_with     = proj_with["interest"].sum()
    int_saved    = int_without - int_with
    months_saved = (dur_without - dur_with) if (dur_without and dur_with) else None

    # Summary cards
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(
            f'<div class="metric-card" style="border-left-color:{COLORS["red"]}">'
            f'<div class="metric-label">Without Prepayment</div>'
            f'<div class="metric-value" style="color:{COLORS["red"]}">'
            f'{months_str(proj_without)}</div>'
            f'<div class="metric-sub">Total interest: {fmt_inr(int_without)}<br>'
            f'Total outflow: {fmt_inr(outstanding + int_without)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with sc2:
        st.markdown(
            f'<div class="metric-card" style="border-left-color:{COLORS["blue"]}">'
            f'<div class="metric-label">With ₹{lump_sum:,} on {prepay_date.strftime("%d %b %Y")}</div>'
            f'<div class="metric-value" style="color:{COLORS["blue"]}">'
            f'{months_str(proj_with)}</div>'
            f'<div class="metric-sub">Total interest: {fmt_inr(int_with)}<br>'
            f'Total outflow: {fmt_inr(outstanding + lump_sum + int_with)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with sc3:
        ms_txt = f"{months_saved} months" if months_saved else "N/A"
        st.markdown(
            f'<div class="metric-card" style="border-left-color:{COLORS["green"]}">'
            f'<div class="metric-label">Time Saved</div>'
            f'<div class="metric-value" style="color:{COLORS["green"]}">{ms_txt}</div>'
            f'<div class="metric-sub">Loan closes earlier<br>'
            f'Prepayment in month {months_to_prepay} (~{prepay_date.strftime("%b %Y")})'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with sc4:
        st.markdown(
            f'<div class="metric-card" style="border-left-color:{COLORS["gold"]}">'
            f'<div class="metric-label">Interest Saved</div>'
            f'<div class="metric-value" style="color:{COLORS["gold"]}">'
            f'{fmt_inr(int_saved)}</div>'
            f'<div class="metric-sub">Net saving after lump sum cost<br>'
            f'Effective return on prepayment: '
            f'{int_saved/lump_sum*100:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Dual balance curve
    st.markdown('<div class="section-header">Outstanding Balance: With vs Without Prepayment</div>',
                unsafe_allow_html=True)

    x_wo = [pd.Timestamp(TODAY) + pd.DateOffset(months=int(mo))
            for mo in proj_without["month"]]
    x_wi = [pd.Timestamp(TODAY) + pd.DateOffset(months=int(mo))
            for mo in proj_with["month"]]

    fig_pp = go.Figure()
    fig_pp.add_trace(go.Scatter(
        x=x_wo, y=proj_without["bal"] / 1e5,
        name=f"No prepayment ({months_str(proj_without)})",
        line=dict(color=COLORS["red"], width=2.5, dash="dash"),
        hovertemplate="%{x|%b %Y}: ₹%{y:.2f}L remaining<extra></extra>",
    ))
    fig_pp.add_trace(go.Scatter(
        x=x_wi, y=proj_with["bal"] / 1e5,
        name=f"With ₹{fmt_inr(lump_sum)} on {prepay_date.strftime('%d %b %Y')} ({months_str(proj_with)})",
        line=dict(color=COLORS["green"], width=2.5),
        fill="tonexty", fillcolor="rgba(39,174,96,0.08)",
        hovertemplate="%{x|%b %Y}: ₹%{y:.2f}L remaining<extra></extra>",
    ))
    # Mark the prepayment date
    prepay_ts = pd.Timestamp(TODAY) + pd.DateOffset(months=months_to_prepay)
    fig_pp.add_vline(
        x=int(prepay_ts.timestamp() * 1000), line_dash="dot",
        line_color=COLORS["gold"], line_width=2,
        annotation_text=f"Prepayment: {fmt_inr(lump_sum)}",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["gold"], size=11),
    )
    fig_pp.add_hline(y=0, line_color=COLORS["navy"], line_width=1, opacity=0.3)
    fig_pp.update_layout(
        height=400, margin=dict(t=10, b=50, l=70, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickformat="%b'%y"),
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE",
                   title="Outstanding (₹ Lakhs)", ticksuffix="L"),
        legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
        hovermode="x unified",
    )
    st.plotly_chart(fig_pp, use_container_width=True)

    # Month-by-month interest comparison
    st.markdown('<div class="section-header">Monthly Interest: Savings Breakdown</div>',
                unsafe_allow_html=True)

    max_mo = max(len(proj_without), len(proj_with))
    mo_idx = list(range(1, max_mo + 1))

    def _pad(series, length):
        arr = list(series)
        return arr + [0.0] * (length - len(arr))

    int_wo_m = _pad(proj_without["interest"], max_mo)
    int_wi_m = _pad(proj_with["interest"], max_mo)
    x_mo     = [pd.Timestamp(TODAY) + pd.DateOffset(months=mo) for mo in mo_idx]

    fig_ms = go.Figure()
    fig_ms.add_trace(go.Scatter(
        x=x_mo, y=int_wo_m,
        name="Interest without prepayment",
        line=dict(color=COLORS["red"], width=2, dash="dash"),
        hovertemplate="%{x|%b %Y}: ₹%{y:,.0f}<extra></extra>",
    ))
    fig_ms.add_trace(go.Scatter(
        x=x_mo, y=int_wi_m,
        name="Interest with prepayment",
        line=dict(color=COLORS["green"], width=2),
        fill="tonexty", fillcolor="rgba(39,174,96,0.10)",
        hovertemplate="%{x|%b %Y}: ₹%{y:,.0f}<extra></extra>",
    ))
    fig_ms.add_vline(
        x=prepay_ts, line_dash="dot", line_color=COLORS["gold"], line_width=2,
    )
    fig_ms.update_layout(
        height=300, margin=dict(t=10, b=50, l=70, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickformat="%b'%y"),
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE",
                   title="Monthly Interest (₹)", tickprefix="₹"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        hovermode="x unified",
    )
    st.plotly_chart(fig_ms, use_container_width=True)

    # Multiple lump-sum comparison table
    st.markdown('<div class="section-header">Compare Different Lump-Sum Amounts on Same Date</div>',
                unsafe_allow_html=True)
    step = max(10_000, int(outstanding // 10 // 10_000) * 10_000)
    test_amounts = sorted(set([
        50_000, 1_00_000, 2_00_000, 3_00_000, 5_00_000,
        lump_sum,
        min(int(outstanding * 0.1), int(outstanding)),
        min(int(outstanding * 0.2), int(outstanding)),
        min(int(outstanding * 0.3), int(outstanding)),
    ]))
    test_amounts = [a for a in test_amounts if 0 < a <= outstanding]
    cmp_rows = []
    for amt in test_amounts:
        pw   = project_with_lumpsum(emi_for_calc, outstanding,
                                    amt, months_to_prepay)
        ti   = pw["interest"].sum()
        cmp_rows.append({
            "Lump Sum":      fmt_inr_full(amt),
            "Payoff":        months_str(pw),
            "Total Interest": fmt_inr_full(ti),
            "Interest Saved": fmt_inr_full(int_without - ti),
            "ROI on Prepayment": f"{(int_without - ti) / amt * 100:.1f}%",
            "Selected":      "✓" if amt == lump_sum else "",
        })
    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
    st.caption("*Net Saving = Interest saved (does not account for opportunity cost of the lump-sum amount)")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 5 — TRANSACTION LOG
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Transaction Log</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        search = st.text_input("🔍 Search remarks",
                               placeholder="e.g. disbursement, NEFT, interest")
    with fc2:
        type_filter = st.multiselect(
            "Filter by type",
            options=["Disbursement", "Interest", "IntPmt", "Payment", "Other"],
            default=["Disbursement", "Interest", "Payment", "IntPmt"],
        )
    with fc3:
        sort_order = st.selectbox("Sort", ["Newest first", "Oldest first"])

    disp = raw_df.copy()
    disp = disp[(disp["Date"] >= filter_start) & (disp["Date"] <= filter_end)]
    if search:
        disp = disp[disp["Remarks"].str.contains(search, case=False, na=False)]
    if type_filter:
        disp = disp[disp["Type"].isin(type_filter)]
    if sort_order == "Newest first":
        disp = disp.sort_values("Date", ascending=False)

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1: st.metric("Shown", len(disp))
    with mc2: st.metric("Total", len(raw_df))
    with mc3: st.metric("Disbursements",
                        len(disp[disp["Type"] == "Disbursement"]))
    with mc4: st.metric("Payments",
                        len(disp[disp["Type"].isin(["Payment", "IntPmt"])]))
    with mc5: st.metric("Manual entries",
                        int((disp["Source"] == "manual").sum())
                        if "Source" in disp.columns else 0)

    out = disp[["Date", "Type", "Remarks", "Amount", "Balance"]].copy()
    if "Source" in disp.columns:
        out["Source"] = disp["Source"]
    out["Date"] = out["Date"].dt.strftime("%d %b %Y")
    out = out.rename(columns={"Type": "Category"})

    COLOR_MAP = {
        "Disbursement": "#FFF3CD",
        "Interest":     "#FDECEA",
        "Payment":      "#D5F5E3",
        "IntPmt":       "#E8F8F5",
        "Other":        "#F5F5F5",
    }

    def style_row(row):
        c = COLOR_MAP.get(row["Category"], "#FFFFFF")
        return [f"background-color:{c}"] * len(row)

    st.dataframe(
        out.style.apply(style_row, axis=1),
        use_container_width=True, height=520,
    )

    csv_data = out.to_csv(index=False)
    st.download_button(
        "⬇️ Download as CSV",
        data=csv_data,
        file_name=f"loan_transactions_{TODAY.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 6 — KEY METRICS
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Key Metrics Summary</div>',
                unsafe_allow_html=True)

    proj_20 = project(20_000, outstanding)
    proj_30 = project(30_000, outstanding)
    proj_50 = project(50_000, outstanding)
    proj_75 = project(75_000, outstanding)
    proj_cu = project(monthly_pmt, outstanding)

    summary = {
        "Metric": [
            "Total Disbursed",
            "Outstanding Balance",
            "Total Repaid (all payments)",
            "Principal Repaid",
            "Interest Charged (Bank)",
            "Theory Interest @8.55%",
            "Interest Delta (Bank − Theory)",
            "Monthly Breakeven (interest only)",
            f"Payoff @ ₹{monthly_pmt:,}/mo (slider)",
            "Payoff @ ₹20,000/mo",
            "Payoff @ ₹30,000/mo",
            "Payoff @ ₹50,000/mo",
            "Payoff @ ₹75,000/mo",
        ],
        "Value": [
            fmt_inr_full(total_disbursed),
            fmt_inr_full(outstanding),
            fmt_inr_full(total_repaid),
            fmt_inr_full(principal_repaid),
            fmt_inr_full(total_int_charged),
            fmt_inr_full(theory_total_int),
            fmt_inr_full(total_int_charged - theory_total_int),
            fmt_inr_full(breakeven),
            months_str(proj_cu),
            months_str(proj_20),
            months_str(proj_30),
            months_str(proj_50),
            months_str(proj_75),
        ],
        "Notes": [
            f"{len(disb)} tranches: {', '.join(d['tag'] for d in disb_meta)}",
            f"As of {TODAY.strftime('%d %b %Y')}",
            f"Total money paid in (incl. interest collections)",
            f"{principal_repaid/total_disbursed*100:.1f}% of total disbursed",
            f"From bank statement ({n_ints} monthly entries)",
            "Daily accrual on actual outstanding from loan start",
            "Positive = bank charged more; Negative = less",
            "Minimum to stop principal growing",
            "Fixed EMI from today — custom scenario",
            "Fixed ₹20K/mo from today",
            "Fixed ₹30K/mo from today",
            "Fixed ₹50K/mo from today",
            "Fixed ₹75K/mo from today",
        ],
    }
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True,
                 height=500)

    st.markdown('<div class="section-header">Interest Savings vs ₹20K/mo Baseline</div>',
                unsafe_allow_html=True)
    base_int = proj_20["interest"].sum()
    sav_rows = []
    for pmt in [30_000, 50_000, 75_000, 1_00_000, monthly_pmt]:
        p = project(pmt, outstanding)
        if p["bal"].iloc[-1] <= 100:
            saved = base_int - p["interest"].sum()
            sav_rows.append({
                "Payment":         f"₹{pmt:,}/mo",
                "Total Interest":  fmt_inr_full(p["interest"].sum()),
                "Saves vs ₹20K":   fmt_inr_full(saved),
                "Payoff Time":     months_str(p),
            })
    if sav_rows:
        st.dataframe(pd.DataFrame(sav_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Loan Progress</div>', unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    with pc1:
        pct = principal_repaid / total_disbursed * 100
        st.progress(pct / 100,
                    text=f"Principal repaid: {pct:.1f}%  "
                         f"({fmt_inr(principal_repaid)} of {fmt_inr(total_disbursed)})")
    with pc2:
        age_pct = min(loan_age_months / 84 * 100, 100)
        st.progress(age_pct / 100,
                    text=f"Loan age: {int(loan_age_months)} months  "
                         f"({age_pct:.0f}% of 7-year horizon)")

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#7F8C8D;font-size:10px;'>"
    f"As on {TODAY.strftime('%d %b %Y')} &nbsp;·&nbsp; "
    f"Rate: {RATE_PA*100:.2f}% p.a. &nbsp;·&nbsp; "
    f"For personal reference only &nbsp;·&nbsp; "
    f"Data refreshes every 5 min"
    f"</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────────────
if _on_loan_page and auto_refresh:
    time.sleep(300)
    st.rerun()
