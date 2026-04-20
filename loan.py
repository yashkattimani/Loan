#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  EDUCATION LOAN ANALYTICS DASHBOARD  —  Union Bank of India
  Account: 374106550610128  |  Student: Yash Kattimani  |  Rate: 8.55% p.a.
  Currency: INR  |  Dashboard Date: 16 Apr 2026
═══════════════════════════════════════════════════════════════════════════════
  Run:  python loan.py
  Output: loan_dashboard_page1.png  +  loan_dashboard_page2.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from datetime import datetime, timedelta, date
import re
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
RATE_PA   = 8.55 / 100
BANK      = "Union Bank of India"
LOAN_ACCT = "374106550610128"
CSV_FILE  = "Loan analyzer.csv"
TODAY     = date(2026, 4, 16)

C = dict(
    navy="#1A2E4A",    blue="#1B5299",   gold="#C8912A",
    green="#27AE60",   red="#C0392B",    orange="#E67E22",
    purple="#8E44AD",  teal="#16A085",   gray="#7F8C8D",
    light="#ECF0F1",   white="#FFFFFF",  bg="#F5F7FA",
    d1="#3498DB",      d2="#2ECC71",     d3="#E74C3C",  d4="#F39C12",
)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False,   "axes.spines.right": False,
    "figure.dpi": 150,          "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fmt_inr(x, _=None):
    ax = abs(x)
    if ax >= 1e7:  return f"Rs{x/1e7:.1f} Cr"
    if ax >= 1e5:  return f"Rs{x/1e5:.1f} L"
    if ax >= 1e3:  return f"Rs{x/1e3:.0f}K"
    return f"Rs{x:.0f}"

def inr(x):
    if x >= 1e7:  return f"Rs {x/1e7:.2f} Cr"
    if x >= 1e5:  return f"Rs {x/1e5:.2f} L"
    return f"Rs {x:,.0f}"

def inr_full(x):
    s = f"{abs(x):,.0f}"
    sign = "-" if x < 0 else ""
    return f"Rs {sign}{s}"

def parse_amt(s):
    if pd.isna(s) or not str(s).strip(): return 0.0
    s   = str(s).strip()
    dr  = "(Dr)" in s
    val = float(re.sub(r"[^\d.]", "", s.replace("(Dr)", "").replace("(Cr)", "")))
    return -val if dr else val

def parse_bal(s):
    if pd.isna(s) or not str(s).strip(): return np.nan
    try:    return float(re.sub(r"[,\s]", "", str(s).strip()))
    except: return np.nan

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    raw = pd.read_csv(CSV_FILE)
    raw.columns = ["Date", "TranId", "Remarks", "Amount",
                   "Balance", "Deposited", "BankInt"]
    raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
    raw["Amt"]  = raw["Amount"].apply(parse_amt)
    raw["Bal"]  = raw["Balance"].apply(parse_bal)
    raw         = raw.sort_values("Date").reset_index(drop=True)
    raw["Abs"]  = raw["Amt"].abs()

    def classify(r):
        rem = str(r["Remarks"]).lower()
        v   = r["Amt"]
        if "loan disbursement" in rem and v < 0: return "Disbursement"
        if re.search(r":n int\.:", rem):         return "Interest"
        if "loan coll" in rem and v > 0:         return "IntPmt"
        if v > 0:                                return "Payment"
        return "Other"

    raw["Type"] = raw.apply(classify, axis=1)
    return raw

df = load_data()

disb  = df[df["Type"] == "Disbursement"].reset_index(drop=True)
ints  = df[df["Type"] == "Interest"].reset_index(drop=True)
pmts  = df[df["Type"] == "Payment"].reset_index(drop=True)
ipmt  = df[df["Type"] == "IntPmt"].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════
total_disbursed   = disb["Abs"].sum()
total_int_charged = ints["Abs"].sum()
total_paid_princ  = pmts["Abs"].sum()
total_paid_int    = ipmt["Abs"].sum()
outstanding       = abs(df["Bal"].dropna().iloc[-1])
principal_repaid  = total_disbursed - outstanding
loan_start        = disb["Date"].min()
loan_age_days     = (pd.Timestamp(TODAY) - loan_start).days
loan_age_months   = loan_age_days / 30.44

bal_ts          = df[df["Bal"].notna()][["Date", "Bal"]].copy()
bal_ts["Outs"]  = bal_ts["Bal"].abs()
bal_ts          = bal_ts.sort_values("Date").reset_index(drop=True)

ints_s          = ints.sort_values("Date").copy()
ints_s["Month"] = ints_s["Date"].dt.to_period("M")
mon_int         = ints_s.groupby("Month")["Abs"].sum().reset_index()
mon_int["Dt"]   = mon_int["Month"].dt.to_timestamp()
ints_s["CumInt"]= ints_s["Abs"].cumsum()

pmts_s          = pmts.sort_values("Date").copy()
pmts_s["Month"] = pmts_s["Date"].dt.to_period("M")
mon_pmt         = pmts_s.groupby("Month")["Abs"].sum().reset_index()
mon_pmt["Dt"]   = mon_pmt["Month"].dt.to_timestamp()
pmts_s["CumPmt"]= pmts_s["Abs"].cumsum()

# ══════════════════════════════════════════════════════════════════════════════
#  THEORETICAL INTEREST  (8.55% daily on actual outstanding)
# ══════════════════════════════════════════════════════════════════════════════
def calc_theory():
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

theory           = calc_theory()
theory_total_int = theory["CumInt"].iloc[-1]

# ══════════════════════════════════════════════════════════════════════════════
#  BACK-CALCULATED RATE  PER PERIOD
# ══════════════════════════════════════════════════════════════════════════════
def parse_int_period(rem):
    m = re.search(r"(\d{2}-\d{2}-\d{4})\s+to\s+(\d{2}-\d{2}-\d{4})", str(rem))
    if m:
        s = datetime.strptime(m.group(1), "%d-%m-%Y")
        e = datetime.strptime(m.group(2), "%d-%m-%Y")
        return s, e, (e - s).days + 1
    return None, None, None

parsed = pd.DataFrame(
    ints_s["Remarks"].apply(parse_int_period).tolist(),
    columns=["P_Start", "P_End", "P_Days"],
    index=ints_s.index,
)
ints_s = pd.concat([ints_s, parsed], axis=1)

implied_rates = []
for _, row in ints_s.iterrows():
    if row["P_Start"] and row["P_Days"]:
        prior_bal = bal_ts[bal_ts["Date"] <= pd.Timestamp(row["P_Start"])]["Outs"]
        if len(prior_bal) > 0:
            b = float(prior_bal.iloc[-1])
            if b > 0 and row["P_Days"] > 0:
                r = row["Abs"] / b * 365 / row["P_Days"] * 100
                implied_rates.append({"Date": row["Date"], "Rate": r})

rate_df = (pd.DataFrame(implied_rates) if implied_rates
           else pd.DataFrame(columns=["Date", "Rate"]))

# ══════════════════════════════════════════════════════════════════════════════
#  DISBURSEMENT METADATA
# ══════════════════════════════════════════════════════════════════════════════
disb_meta = [
    dict(n=1, tag="Jul'23", dt=disb.iloc[0]["Date"].date(),
         amt=disb.iloc[0]["Abs"], col=C["d1"]),
    dict(n=2, tag="Jan'24", dt=disb.iloc[1]["Date"].date(),
         amt=disb.iloc[1]["Abs"], col=C["d2"]),
    dict(n=3, tag="Aug'24", dt=disb.iloc[2]["Date"].date(),
         amt=disb.iloc[2]["Abs"], col=C["d3"]),
    dict(n=4, tag="Feb'25", dt=disb.iloc[3]["Date"].date(),
         amt=disb.iloc[3]["Abs"], col=C["d4"]),
]
for d in disb_meta:
    d["days"]       = (TODAY - d["dt"]).days
    d["theory_int"] = d["amt"] * RATE_PA / 365 * d["days"]

# ══════════════════════════════════════════════════════════════════════════════
#  PROJECTION
# ══════════════════════════════════════════════════════════════════════════════
def project(monthly_pmt, max_months=360):
    r   = RATE_PA / 12
    bal = outstanding
    rows = []
    for m in range(1, max_months + 1):
        interest  = bal * r
        principal = max(0.0, monthly_pmt - interest)
        bal      -= principal
        rows.append(dict(month=m, bal=max(0.0, bal),
                         interest=interest, principal=principal))
        if bal <= 0:
            break
    return pd.DataFrame(rows)

def months_str(df_):
    if df_["bal"].iloc[-1] > 100: return "Never (underpays)"
    y, m = divmod(len(df_), 12)
    return (f"{y}y {m}m" if y else f"{m}m")

proj_20 = project(20_000)
proj_30 = project(30_000)
proj_50 = project(50_000)
proj_75 = project(75_000)

# ══════════════════════════════════════════════════════════════════════════════
#  KPI CARD HELPER
# ══════════════════════════════════════════════════════════════════════════════
def kpi_card(ax, title, value, sub="", col=C["blue"]):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.03",
        facecolor=col + "22", edgecolor=col, linewidth=2))
    ax.add_patch(FancyBboxPatch(
        (0.05, 0.87), 0.9, 0.08, boxstyle="round,pad=0.01",
        facecolor=col, edgecolor="none"))
    ax.text(0.5, 0.72, title,  ha="center", va="center",
            fontsize=7.5, color=C["navy"], fontweight="bold")
    ax.text(0.5, 0.48, value,  ha="center", va="center",
            fontsize=12,  color=col,        fontweight="bold")
    if sub:
        ax.text(0.5, 0.22, sub, ha="center", va="center",
                fontsize=7, color=C["gray"])


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1  —  MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(20, 13), facecolor=C["bg"])
gs1  = gridspec.GridSpec(5, 6, figure=fig1,
                         hspace=0.58, wspace=0.45,
                         top=0.90, bottom=0.06,
                         left=0.05, right=0.97)

# ─── Header ──────────────────────────────────────────────────────────────────
fig1.text(0.5, 0.955,
          "EDUCATION LOAN ANALYTICS DASHBOARD",
          ha="center", fontsize=16, fontweight="bold", color=C["navy"])
fig1.text(0.5, 0.935,
          f"{BANK}  |  Account: {LOAN_ACCT}  |  "
          f"Rate: 8.55% p.a.  |  As on: {TODAY.strftime('%d %b %Y')}",
          ha="center", fontsize=9, color=C["gray"])

# ─── Thin accent bar ─────────────────────────────────────────────────────────
fig1.add_artist(
    plt.Line2D([0.05, 0.95], [0.928, 0.928],
               transform=fig1.transFigure,
               color=C["gold"], linewidth=2.5))

# ─── KPI cards ───────────────────────────────────────────────────────────────
kpi_data = [
    ("TOTAL DISBURSED",     inr(total_disbursed),
     f"4 tranches over 2 yrs",    C["blue"]),
    ("OUTSTANDING",         inr(outstanding),
     f"Loan balance today",        C["red"]),
    ("PRINCIPAL REPAID",    inr(principal_repaid),
     f"{principal_repaid/total_disbursed*100:.1f}% of disbursed", C["green"]),
    ("INTEREST CHARGED",    inr(total_int_charged),
     f"Actual (bank statement)",   C["orange"]),
    ("LOAN AGE",            f"{int(loan_age_months)} months",
     f"{loan_age_days} days since Jul 2023", C["purple"]),
    ("THEORY INT @8.55%",   inr(theory_total_int),
     "Daily accrual on balance",   C["teal"]),
]
for i, (ttl, val, sub, col) in enumerate(kpi_data):
    kpi_card(fig1.add_subplot(gs1[0, i]), ttl, val, sub, col)

# ─── Outstanding Balance Line ─────────────────────────────────────────────────
ax_bal = fig1.add_subplot(gs1[1:3, :4])
ax_bal.set_facecolor(C["white"])
ax_bal.fill_between(bal_ts["Date"], 0, bal_ts["Outs"],
                    alpha=0.12, color=C["blue"])
ax_bal.plot(bal_ts["Date"], bal_ts["Outs"],
            color=C["blue"], linewidth=2.4, zorder=3)

for i, row in disb.iterrows():
    bal_at = bal_ts[bal_ts["Date"] <= row["Date"]]["Outs"]
    y      = float(bal_at.iloc[-1]) if len(bal_at) else 0
    col    = [C["d1"], C["d2"], C["d3"], C["d4"]][i]
    ax_bal.axvline(row["Date"], color=col,
                   linestyle="--", alpha=0.55, linewidth=1.5, zorder=2)
    ax_bal.scatter([row["Date"]], [y],
                   color=col, s=90, zorder=5,
                   edgecolors="white", linewidth=1.5)
    ax_bal.text(row["Date"], y + 55_000,
                f"D{i+1}\n{inr(row['Abs'])}",
                ha="center", va="bottom", fontsize=6.5,
                color=col, fontweight="bold")

ax_bal.yaxis.set_major_formatter(FuncFormatter(fmt_inr))
ax_bal.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax_bal.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax_bal.xaxis.get_majorticklabels(),
         rotation=45, ha="right", fontsize=7.5)
ax_bal.set_title("Outstanding Loan Balance Over Time",
                 fontsize=11, fontweight="bold", color=C["navy"], pad=8)
ax_bal.set_ylabel("Outstanding (Rs)", fontsize=8.5, color=C["gray"])
ax_bal.set_ylim(bottom=0)
ax_bal.grid(axis="y", linestyle="--", alpha=0.35)
ax_bal.tick_params(labelsize=8)

# ─── Disbursement Donut ──────────────────────────────────────────────────────
ax_pie = fig1.add_subplot(gs1[1:3, 4:])
ax_pie.set_facecolor(C["white"])

pie_sizes  = [d["amt"]  for d in disb_meta]
pie_colors = [d["col"]  for d in disb_meta]
pie_labels = [f"T{d['n']} {d['tag']}\n{inr(d['amt'])}" for d in disb_meta]

wedges, texts, autotexts = ax_pie.pie(
    pie_sizes, labels=None,
    colors=pie_colors, autopct="%1.1f%%",
    startangle=140, explode=[0.04] * 4,
    pctdistance=0.75,
    wedgeprops=dict(width=0.62, edgecolor="white", linewidth=2),
)
for at in autotexts:
    at.set_fontsize(7.5); at.set_fontweight("bold"); at.set_color(C["white"])
ax_pie.text(0, 0, f"Total\n{inr(total_disbursed)}",
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=C["navy"])
legend_p = [
    mpatches.Patch(color=d["col"],
                   label=f"Tranche {d['n']} ({d['tag']}): {inr(d['amt'])}")
    for d in disb_meta
]
ax_pie.legend(handles=legend_p, loc="lower center",
              bbox_to_anchor=(0.5, -0.20), ncol=2,
              fontsize=7.5, frameon=False)
ax_pie.set_title("College Fee Disbursements\n(4 Tranches — Education Loan)",
                 fontsize=10, fontweight="bold", color=C["navy"], pad=8)

# ─── Monthly Interest Bar ─────────────────────────────────────────────────────
ax_mi  = fig1.add_subplot(gs1[3:5, :3])
ax_mi2 = ax_mi.twinx()
ax_mi.set_facecolor(C["white"])

ax_mi.bar(mon_int["Dt"], mon_int["Abs"],
          color=C["orange"], alpha=0.75, width=25, zorder=3)
ax_mi2.plot(mon_int["Dt"], mon_int["Abs"].cumsum() / 1e5,
            color=C["red"], linewidth=2, marker=".", markersize=4, zorder=4)
ax_mi2.set_ylabel("Cumulative (Rs L)", fontsize=7.5, color=C["red"])
ax_mi2.tick_params(axis="y", colors=C["red"], labelsize=7.5)
ax_mi2.spines["right"].set_color(C["red"])

ax_mi.yaxis.set_major_formatter(FuncFormatter(fmt_inr))
ax_mi.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax_mi.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax_mi.xaxis.get_majorticklabels(),
         rotation=45, ha="right", fontsize=7)
ax_mi.set_title("Monthly Interest Charged by Bank  (Bars = monthly | Line = cumulative)",
                fontsize=10, fontweight="bold", color=C["navy"], pad=6)
ax_mi.set_ylabel("Monthly Interest (Rs)", fontsize=8, color=C["gray"])
ax_mi.grid(axis="y", linestyle="--", alpha=0.3)
ax_mi.tick_params(labelsize=7.5)
ax_mi.set_ylim(bottom=0)

# ─── Cumulative: Interest vs Principal repaid ─────────────────────────────────
ax_cum = fig1.add_subplot(gs1[3:5, 3:])
ax_cum.set_facecolor(C["white"])

ax_cum.fill_between(ints_s["Date"], 0, ints_s["CumInt"] / 1e5,
                    alpha=0.20, color=C["orange"])
ax_cum.plot(ints_s["Date"], ints_s["CumInt"] / 1e5,
            color=C["orange"], linewidth=2.2,
            label=f"Interest Charged (Total: {inr(total_int_charged)})")

ax_cum.plot(theory["Date"], theory["CumInt"] / 1e5,
            color=C["red"], linewidth=1.6, linestyle="--",
            label=f"Theoretical @8.55% ({inr(theory_total_int)})")

ax_cum.fill_between(pmts_s["Date"], 0, pmts_s["CumPmt"] / 1e5,
                    alpha=0.20, color=C["green"])
ax_cum.plot(pmts_s["Date"], pmts_s["CumPmt"] / 1e5,
            color=C["green"], linewidth=2.2,
            label=f"Principal Payments ({inr(total_paid_princ)})")

ax_cum.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"Rs {x:.0f}L"))
ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax_cum.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax_cum.xaxis.get_majorticklabels(),
         rotation=45, ha="right", fontsize=7)
ax_cum.set_title("Cumulative Interest Charged  vs  Principal Repaid",
                 fontsize=10, fontweight="bold", color=C["navy"], pad=6)
ax_cum.set_ylabel("Rs Lakhs", fontsize=8, color=C["gray"])
ax_cum.legend(fontsize=7.5, frameon=False, loc="upper left")
ax_cum.grid(axis="y", linestyle="--", alpha=0.3)
ax_cum.tick_params(labelsize=7.5)
ax_cum.set_ylim(bottom=0)

fig1.text(0.5, 0.015,
          f"Generated on {TODAY.strftime('%d %b %Y')}  |  "
          "Interest @ 8.55% p.a. on actual outstanding balance  |  For personal reference only",
          ha="center", fontsize=7.5, color=C["gray"])

fig1.savefig("loan_dashboard_page1.png", facecolor=C["bg"])
print("  Page 1 saved  ->  loan_dashboard_page1.png")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2  —  DEEP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(20, 14), facecolor=C["bg"])
gs2  = gridspec.GridSpec(5, 6, figure=fig2,
                         hspace=0.62, wspace=0.50,
                         top=0.91, bottom=0.06,
                         left=0.05, right=0.97)

fig2.text(0.5, 0.953,
          "EDUCATION LOAN — DEEP ANALYSIS",
          ha="center", fontsize=15, fontweight="bold", color=C["navy"])
fig2.text(0.5, 0.933,
          f"{BANK}  |  {inr_full(total_disbursed)} disbursed  |  "
          f"{inr_full(outstanding)} outstanding  |  "
          f"Rate 8.55% p.a.  |  As on {TODAY.strftime('%d %b %Y')}",
          ha="center", fontsize=8.5, color=C["gray"])
fig2.add_artist(
    plt.Line2D([0.05, 0.95], [0.928, 0.928],
               transform=fig2.transFigure,
               color=C["gold"], linewidth=2.5))

# ─── Disbursement Timeline ────────────────────────────────────────────────────
ax_tl = fig2.add_subplot(gs2[0, :])
ax_tl.set_xlim(0, 1); ax_tl.set_ylim(0, 1); ax_tl.axis("off")
ax_tl.set_facecolor(C["white"])
ax_tl.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0",
                                facecolor=C["white"], edgecolor="none",
                                transform=ax_tl.transAxes))
ax_tl.axhline(0.50, 0.02, 0.98, color=C["navy"],
              linewidth=3, alpha=0.30, solid_capstyle="round")
ax_tl.text(0.01, 0.92, "DISBURSEMENT TIMELINE",
           fontsize=10, fontweight="bold", color=C["navy"], va="top")

ref_start   = loan_start
total_span  = (pd.Timestamp("2026-07-01") - ref_start).days

for d in disb_meta:
    x = (pd.Timestamp(d["dt"]) - ref_start).days / total_span
    ax_tl.plot([x], [0.50], "o",
               markersize=20, color=d["col"],
               markeredgecolor="white", markeredgewidth=2.5, zorder=5)
    ax_tl.text(x, 0.50, str(d["n"]),
               ha="center", va="center", fontsize=9,
               fontweight="bold", color="white", zorder=6)
    above = (d["n"] % 2 == 1)
    y_lbl, y_sub = (0.82, 0.70) if above else (0.20, 0.32)
    lineseg_y    = (0.67, 0.56) if above else (0.33, 0.44)
    ax_tl.annotate("", xy=(x, lineseg_y[1]), xytext=(x, lineseg_y[0]),
                   arrowprops=dict(arrowstyle="-", color=d["col"],
                                   lw=1.5, linestyle=":"))
    ax_tl.text(x, y_lbl,           f"Tranche {d['n']}",
               ha="center", fontsize=8.5, fontweight="bold", color=d["col"])
    ax_tl.text(x, y_lbl - 0.10,    d["tag"],
               ha="center", fontsize=8,   color=C["gray"])
    ax_tl.text(x, y_lbl - 0.21,    inr(d["amt"]),
               ha="center", fontsize=9,   fontweight="bold", color=C["navy"])

x_today = (pd.Timestamp(TODAY) - ref_start).days / total_span
ax_tl.axvline(x_today, 0.10, 0.90, color=C["gold"],
              linewidth=2.5, linestyle="--", alpha=0.9)
ax_tl.text(x_today, 0.93, f"Today ({TODAY.strftime('%d %b %Y')})",
           ha="center", va="top", fontsize=7.5,
           color=C["gold"], fontweight="bold")

for yr in [2024, 2025, 2026]:
    xd = (pd.Timestamp(f"{yr}-01-01") - ref_start).days / total_span
    if 0 < xd < 1:
        ax_tl.axvline(xd, 0.40, 0.60, color=C["gray"],
                      linewidth=0.8, linestyle=":", alpha=0.5)
        ax_tl.text(xd, 0.42, str(yr), ha="center",
                   fontsize=7, color=C["gray"])

# ─── Per-Tranche Principal vs Theoretical Interest ────────────────────────────
ax_da = fig2.add_subplot(gs2[1:3, :3])
ax_da.set_facecolor(C["white"])

d_amts  = [d["amt"]        / 1e5 for d in disb_meta]
d_tints = [d["theory_int"] / 1e5 for d in disb_meta]
d_cols  = [d["col"]               for d in disb_meta]
x_pos   = np.arange(len(disb_meta))

bars1 = ax_da.bar(x_pos - 0.22, d_amts,  width=0.42,
                  color=d_cols, alpha=0.70, label="Disbursed Principal")
bars2 = ax_da.bar(x_pos + 0.22, d_tints, width=0.42,
                  color=d_cols, alpha=1.00, hatch="///",
                  edgecolor="white", label="Interest Accrued @8.55% (no payments)")

for b in bars1:
    ax_da.text(b.get_x() + b.get_width()/2, b.get_height() + 0.25,
               f"Rs {b.get_height():.1f}L", ha="center", va="bottom",
               fontsize=6.5, fontweight="bold", color=C["navy"])
for b in bars2:
    ax_da.text(b.get_x() + b.get_width()/2, b.get_height() + 0.25,
               f"Rs {b.get_height():.1f}L", ha="center", va="bottom",
               fontsize=6.5, fontweight="bold", color=C["red"])

ax_da.set_xticks(x_pos)
ax_da.set_xticklabels(
    [f"Tranche {d['n']}\n{d['tag']}\n({d['days']} days)" for d in disb_meta],
    fontsize=7.5)
ax_da.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"Rs {x:.0f}L"))
ax_da.set_title(
    "Per-Tranche: Principal Disbursed  vs  Interest Cost @ 8.55%\n"
    "(*Gross interest if zero payments — illustrative cost per tranche)",
    fontsize=10, fontweight="bold", color=C["navy"], pad=6)
ax_da.set_ylabel("Rs Lakhs", fontsize=8, color=C["gray"])
ax_da.legend(fontsize=8, frameon=False)
ax_da.grid(axis="y", linestyle="--", alpha=0.3)
ax_da.tick_params(labelsize=8)
ax_da.set_ylim(bottom=0)

# ─── Back-Calculated Effective Rate ──────────────────────────────────────────
ax_ir = fig2.add_subplot(gs2[1:3, 3:])
ax_ir.set_facecolor(C["white"])

if len(rate_df) > 0:
    ax_ir.plot(rate_df["Date"], rate_df["Rate"],
               color=C["purple"], linewidth=1.8,
               marker="D", markersize=5, zorder=4,
               label="Effective rate per period")
    ax_ir.fill_between(rate_df["Date"], RATE_PA * 100, rate_df["Rate"],
                       where=rate_df["Rate"] >= RATE_PA * 100,
                       alpha=0.25, color=C["orange"],
                       label=f"Above {RATE_PA*100:.2f}%")
    ax_ir.fill_between(rate_df["Date"], rate_df["Rate"], RATE_PA * 100,
                       where=rate_df["Rate"] < RATE_PA * 100,
                       alpha=0.20, color=C["green"],
                       label=f"Below {RATE_PA*100:.2f}%")

ax_ir.axhline(RATE_PA * 100, color=C["blue"], linewidth=2.2,
              linestyle="--", label="8.55% reference")
ax_ir.text(rate_df["Date"].iloc[-1] if len(rate_df) else pd.Timestamp(TODAY),
           RATE_PA * 100 + 0.1, "  8.55%",
           va="bottom", fontsize=7.5, color=C["blue"], fontweight="bold")
for lvl in [8.0, 9.0, 10.0]:
    ax_ir.axhline(lvl, color=C["gray"], linewidth=0.7,
                  linestyle=":", alpha=0.5)
    ax_ir.text(rate_df["Date"].min() if len(rate_df) else pd.Timestamp(loan_start),
               lvl + 0.05, f" {lvl}%",
               fontsize=6.5, color=C["gray"])

ax_ir.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax_ir.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax_ir.xaxis.get_majorticklabels(),
         rotation=45, ha="right", fontsize=7)
ax_ir.set_title(
    "Effective Interest Rate Per Period\n"
    "(Back-calculated from bank charges — shows if rate differs from 8.55%)",
    fontsize=10, fontweight="bold", color=C["navy"], pad=6)
ax_ir.set_ylabel("Effective Annual Rate (%)", fontsize=8, color=C["gray"])
ax_ir.legend(fontsize=7.5, frameon=False, loc="upper right")
ax_ir.grid(axis="y", linestyle="--", alpha=0.3)
ax_ir.tick_params(labelsize=7.5)
if len(rate_df):
    pad = 1.2
    ax_ir.set_ylim(max(0, rate_df["Rate"].min() - pad),
                   rate_df["Rate"].max() + pad)

# ─── Monthly: Interest + Payments stacked ────────────────────────────────────
ax_mp = fig2.add_subplot(gs2[3, :3])
ax_mp.set_facecolor(C["white"])

all_months = pd.period_range(
    df["Date"].min().to_period("M"),
    pd.Timestamp(TODAY).to_period("M"), freq="M")
mi_map  = mon_int.set_index("Month")["Abs"].to_dict()
mp_map  = mon_pmt.set_index("Month")["Abs"].to_dict()
m_dates = [m.to_timestamp() for m in all_months]
m_int   = [mi_map.get(m, 0) / 1e3 for m in all_months]
m_princ = [mp_map.get(m, 0) / 1e3 for m in all_months]

ax_mp.bar(m_dates, m_int,   width=25, color=C["orange"],
          alpha=0.80, label="Interest Charged (Bank)", zorder=3)
ax_mp.bar(m_dates, m_princ, width=25, color=C["green"],
          alpha=0.80, bottom=m_int,
          label="Principal Payments Made", zorder=3)

ax_mp.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"Rs {x:.0f}K"))
ax_mp.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax_mp.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax_mp.xaxis.get_majorticklabels(),
         rotation=45, ha="right", fontsize=7)
ax_mp.set_title("Monthly Breakdown: Interest Charged + Principal Payments",
                fontsize=10, fontweight="bold", color=C["navy"], pad=6)
ax_mp.set_ylabel("Rs Thousands", fontsize=8, color=C["gray"])
ax_mp.legend(fontsize=8, frameon=False)
ax_mp.grid(axis="y", linestyle="--", alpha=0.3)
ax_mp.tick_params(labelsize=7.5)
ax_mp.set_ylim(bottom=0)

# ─── Repayment Projection ─────────────────────────────────────────────────────
ax_proj = fig2.add_subplot(gs2[3, 3:])
ax_proj.set_facecolor(C["white"])

for proj, lbl, col, ls in [
    (proj_20, "Rs 20K/mo", C["red"],    "--"),
    (proj_30, "Rs 30K/mo", C["orange"], "-"),
    (proj_50, "Rs 50K/mo", C["blue"],   "-"),
    (proj_75, "Rs 75K/mo", C["green"],  "-"),
]:
    x_proj = [pd.Timestamp(TODAY) + pd.DateOffset(months=int(m))
              for m in proj["month"]]
    ax_proj.plot(x_proj, proj["bal"] / 1e5,
                 color=col, linewidth=2.2, linestyle=ls,
                 label=f"{lbl} ({months_str(proj)})")

ax_proj.axhline(0, color=C["navy"], linewidth=1, alpha=0.4)
ax_proj.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"Rs {x:.0f}L"))
ax_proj.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax_proj.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax_proj.xaxis.get_majorticklabels(),
         rotation=45, ha="right", fontsize=7)
ax_proj.set_title(
    "Repayment Projection  (Fixed Monthly Payment Scenarios)\n"
    f"Current Outstanding: {inr(outstanding)}  |  Breakeven: {inr(outstanding*RATE_PA/12)}/mo",
    fontsize=10, fontweight="bold", color=C["navy"], pad=6)
ax_proj.set_ylabel("Outstanding (Rs Lakhs)", fontsize=8, color=C["gray"])
ax_proj.legend(fontsize=8, frameon=False, loc="upper right")
ax_proj.grid(axis="y", linestyle="--", alpha=0.3)
ax_proj.tick_params(labelsize=7.5)
ax_proj.set_ylim(bottom=0)

# ─── Summary Table ────────────────────────────────────────────────────────────
ax_tbl = fig2.add_subplot(gs2[4, :])
ax_tbl.axis("off")
ax_tbl.set_facecolor(C["bg"])

table_rows = [
    ["Metric",              "Value",                         "Notes / Detail"],
    ["Total Disbursed",     inr_full(total_disbursed),
     "4 tranches: Jul 2023, Jan 2024, Aug 2024, Feb 2025"],
    ["Outstanding Balance", inr_full(outstanding),
     f"As of {TODAY.strftime('%d %b %Y')}"],
    ["Principal Repaid",    inr_full(principal_repaid),
     f"{principal_repaid/total_disbursed*100:.1f}% of total disbursed"],
    ["Interest Charged",    inr_full(total_int_charged),
     f"From bank statement ({len(ints)} monthly entries)"],
    ["Theory Int @8.55%",   inr_full(theory_total_int),
     "Daily accrual on actual outstanding from Jul 2023"],
    ["Monthly Breakeven",   inr_full(outstanding * RATE_PA / 12),
     "Minimum payment to stop principal from growing"],
    ["Payoff @ Rs 30K/mo",  months_str(proj_30),
     "Fixed payment from today"],
    ["Payoff @ Rs 50K/mo",  months_str(proj_50),
     "Fixed payment from today"],
    ["Payoff @ Rs 75K/mo",  months_str(proj_75),
     "Fixed payment from today"],
]

col_xs  = [0.01, 0.28, 0.52]
row_h   = 1.0 / len(table_rows)

for r_i, row in enumerate(table_rows):
    y   = 1.0 - r_i * row_h
    bg  = C["navy"] if r_i == 0 else (C["light"] if r_i % 2 == 0 else C["white"])
    ax_tbl.add_patch(FancyBboxPatch(
        (0.0, y - row_h + 0.004), 1.0, row_h - 0.005,
        boxstyle="square,pad=0", transform=ax_tbl.transAxes,
        facecolor=bg, edgecolor="none", clip_on=False))
    for c_i, (cell, xstart) in enumerate(zip(row, col_xs)):
        tc = C["white"] if r_i == 0 else C["navy"]
        fw = "bold" if r_i == 0 or c_i == 0 else "normal"
        ax_tbl.text(xstart + 0.01, y - row_h / 2,
                    cell, ha="left", va="center",
                    fontsize=8, color=tc, fontweight=fw,
                    transform=ax_tbl.transAxes)

ax_tbl.set_xlim(0, 1); ax_tbl.set_ylim(0, 1)
ax_tbl.set_title("  KEY METRICS SUMMARY",
                 fontsize=9, fontweight="bold",
                 color=C["navy"], loc="left", pad=3)

fig2.text(
    0.5, 0.015,
    "* Theoretical interest = 8.55% p.a. daily on actual balance from bank statement.  "
    "Back-calculated rates may differ due to bank's own day-count convention or slight "
    "rate changes.  For personal reference only.",
    ha="center", fontsize=7, color=C["gray"])

fig2.savefig("loan_dashboard_page2.png", facecolor=C["bg"])
print("  Page 2 saved  ->  loan_dashboard_page2.png")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
SEP = "=" * 66
sep = "-" * 66
print("\n" + SEP)
print("  EDUCATION LOAN SUMMARY  —  Union Bank of India")
print(SEP)
print(f"  Loan Account  :  {LOAN_ACCT}")
print(f"  Interest Rate :  8.55% p.a. (current)")
print(f"  Loan Start    :  {loan_start.strftime('%d %b %Y')}")
print(f"  Dashboard On  :  {TODAY.strftime('%d %b %Y')}")
print(f"  Loan Age      :  {int(loan_age_months)} months ({loan_age_days} days)")
print(sep)
print(f"  {'Tranche':<10}  {'Date':<14}  {'Amount':<22}  Days Since")
print(sep)
for d in disb_meta:
    print(f"  Tranche {d['n']:<3}  "
          f"{d['tag']:<14}  "
          f"{inr_full(d['amt']):<22}  "
          f"{d['days']} days")
print(sep)
print(f"  {'TOTAL':<10}  {'':14}  {inr_full(total_disbursed)}")
print(sep)
print(f"  Outstanding Balance   :  {inr_full(outstanding)}")
print(f"  Principal Repaid      :  {inr_full(principal_repaid)}"
      f"  ({principal_repaid/total_disbursed*100:.1f}%)")
print(f"  Interest Charged(Bank):  {inr_full(total_int_charged)}")
print(f"  Theory Interest @8.55%:  {inr_full(theory_total_int)}")
print(f"  Monthly interest (now):  {inr_full(outstanding * RATE_PA / 12)}")
print(sep)
print("  PAYOFF PROJECTIONS (from today at fixed monthly payment):")
for proj, lbl in [(proj_20, "Rs 20K/mo"),
                  (proj_30, "Rs 30K/mo"),
                  (proj_50, "Rs 50K/mo"),
                  (proj_75, "Rs 75K/mo")]:
    print(f"    {lbl:14}  -->  {months_str(proj)}")
print(SEP)
print("\n  Dashboard complete!  Open the two PNG files to view.\n")
