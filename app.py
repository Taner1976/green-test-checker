import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from gtc_core import (
    norm_choice,
    parse_key,
    kr20,
    corrected_rpbis,
    upper_lower_groups,
    difficulty_category,
    discrim_category,
    score_distribution_interpretation,
    unidimensionality_pca,
    detect_options,
)

# =========================
# Decision & Notes
# =========================
def item_notes(r, p):
    if pd.isna(r) or pd.isna(p):
        return "Insufficient data."
    if r < 0:
        return "Negative rpbis -> check answer key / ambiguity / miskey risk."
    if r < 0.20:
        if p < 0.20:
            return "Low discrim + very difficult -> difficulty may be driving low rpbis."
        if p > 0.80:
            return "Low discrim + very easy -> too easy; distractors likely weak."
        return "Low discrim + moderate difficulty -> distractors/wording likely issue."
    if 0.20 <= r < 0.30:
        return "Borderline discrim -> review distractors and clarity."
    if 0.30 <= r < 0.40:
        return "Good discrim -> keep; monitor distractors."
    return "Very good discrim -> strong item."

def decision_rule(r, p, n_flags):
    if pd.isna(r) or pd.isna(p):
        return "REVIEW"
    if r < 0:
        return "CHECK KEY"
    if r < 0.20:
        return "REVISE"
    if 0.20 <= r < 0.30:
        return "MINOR REVISE" if n_flags >= 2 else "REVIEW"
    return "KEEP" if n_flags < 3 else "REVIEW DISTRACTORS"

# =========================
# Simple PDF report
# =========================
def build_pdf_bytes(summary_lines, item_table: pd.DataFrame, distr_df: pd.DataFrame):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "GTC Report")
    y -= 25

    c.setFont("Helvetica", 10)
    for line in summary_lines:
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        c.drawString(50, y, str(line)[:120])
        y -= 14

    # Item table (first rows)
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Item Analysis (top rows)")
    y -= 18
    c.setFont("Helvetica", 9)

    it = item_table.copy()
    cols = ["Item", "p (Difficulty)", "rpbis (Corrected)", "Decision", "#DistractorFlags"]
    cols = [col for col in cols if col in it.columns]
    it = it[cols].head(25)

    for _, row in it.iterrows():
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 9)
        line = " | ".join([str(row[col]) for col in cols])
        c.drawString(50, y, line[:120])
        y -= 12

    # Distractor flags (first rows)
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Distractor Flags (top rows)")
    y -= 18
    c.setFont("Helvetica", 9)

    if distr_df is None or distr_df.empty:
        c.drawString(50, y, "No distractor problems found.")
        y -= 12
    else:
        dd = distr_df.head(30)
        for _, r in dd.iterrows():
            if y < 80:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 9)
            line = f"{r['Item']} | {r['Distractor']} | ov={r['Overall %']} up={r['Upper %']} low={r['Lower %']} | {r['Flags']}"
            c.drawString(50, y, line[:120])
            y -= 12

    c.save()
    buf.seek(0)
    return buf.read()

# =========================
# Page
# =========================
st.set_page_config(page_title="GTC - Green Test Checker", layout="wide")
st.title("GTC - Green Test Checker")
st.caption("Sustainable assessment decisions | Diagnose -> Revise -> Reuse")

uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Upload an Excel file to begin.")
    st.stop()

df = pd.read_excel(uploaded)
df.columns = df.columns.astype(str).str.strip()

st.subheader("Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# =========================
# Column mapping
# =========================
st.subheader("Column Mapping")
all_cols = list(df.columns)
default_student = all_cols[:3] if len(all_cols) >= 3 else all_cols

student_cols = st.multiselect(
    "Select student info columns (e.g., ID, Name, Surname)",
    options=all_cols,
    default=default_student
)
item_cols = [c for c in all_cols if c not in student_cols]

if len(item_cols) == 0:
    st.error("No item columns left. Please select student columns correctly.")
    st.stop()

st.write(f"Student columns: {student_cols}")
st.write(f"Item columns detected: {len(item_cols)}")

# =========================
# Answer Key
# =========================
st.subheader("Answer Key")
key_text = st.text_area(
    "Enter answer key (Examples: 'B C A D ...' OR 'BCAD...' OR 'Item1=B, Item2=C, ...')",
    height=90
)
if key_text.strip() == "":
    st.warning("Enter the answer key to proceed.")
    st.stop()

key = parse_key(key_text, item_cols)
if key is None:
    st.error("Answer key length does not match the number of detected item columns.")
    st.stop()
if key.isna().any():
    st.error("Answer key contains missing/unreadable values. Please check formatting.")
    st.stop()

# =========================
# Responses + 0/1
# =========================
resp = df[item_cols].applymap(norm_choice)

st.info("Note: Missing responses are currently treated as incorrect (0). (MVP setting)")

X01 = resp.eq(key, axis=1).astype(float)
X01 = X01.where(~resp.isna(), np.nan).fillna(0).astype(int)

total = X01.sum(axis=1)
k_items = len(item_cols)

# =========================
# Test-level summary
# =========================
st.subheader("Test-Level Summary")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("N", int(len(total)))
c2.metric("Mean", f"{float(total.mean()):.2f}")
c3.metric("SD", f"{float(total.std(ddof=1)):.2f}" if len(total) > 1 else "0.00")
c4.metric("Min-Max", f"{int(total.min())}-{int(total.max())}")
kr = kr20(X01)
c5.metric("KR-20", f"{kr:.3f}" if pd.notna(kr) else "N/A")

st.write(score_distribution_interpretation(total, k_items))

# Histogram
fig, ax = plt.subplots()
bins = min(20, max(8, int(np.sqrt(len(total)))))
ax.hist(total, bins=bins)
ax.set_xlabel("Total Score")
ax.set_ylabel("Number of Students")
ax.set_title("Histogram of Total Scores")
st.pyplot(fig)
plt.close(fig)

# Unidimensionality + Scree
eigvals, uni_msg, var1 = unidimensionality_pca(X01)
st.subheader("Unidimensionality Check (PCA on item correlations)")
st.write(uni_msg)

with st.expander("Scree Plot"):
    fig2, ax2 = plt.subplots()
    x = np.arange(1, min(len(eigvals), 20) + 1)
    ax2.plot(x, eigvals[:len(x)], marker="o")
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Eigenvalue")
    ax2.set_title("Scree Plot (first components)")
    st.pyplot(fig2)
    plt.close(fig2)

# =========================
# Groups for distractors
# =========================
upper_idx, lower_idx = upper_lower_groups(total, frac=0.27)

# =========================
# 1) ITEM ANALYSIS FIRST
# =========================
st.subheader("1) Item Analysis")

p = X01.mean(axis=0)
rpbis = corrected_rpbis(X01)

min_pct = st.slider(
    "Distractor threshold for flagging (overall %) - used in both tables",
    1, 10, 5
)

# Options control: Auto-detect OR manual 2-5
mode = st.radio("Option set", ["Auto-detect from data", "Set manually (2-5)"], horizontal=True)

if mode == "Auto-detect from data":
    options = detect_options(resp[item_cols])
else:
    opt_count = st.selectbox("Number of options per item", [2, 3, 4, 5], index=2)
    options = list("ABCDE")[:opt_count]

st.write("Options used:", options)

# Build distractor flags in-memory (for Decision)
distr_rows = []
if upper_idx is not None and lower_idx is not None:
    for item in item_cols:
        correct = key[item]
        overall = resp[item].value_counts(dropna=True, normalize=True)
        upper = resp.loc[upper_idx, item].value_counts(dropna=True, normalize=True)
        lower = resp.loc[lower_idx, item].value_counts(dropna=True, normalize=True)

        for opt in options:
            if opt == correct:
                continue

            ov = float(overall.get(opt, 0.0))
            up = float(upper.get(opt, 0.0))
            lo = float(lower.get(opt, 0.0))

            flags = []
            if ov < (min_pct / 100.0):
                flags.append(f"Non-functional (<{min_pct}% overall)")
            if ov > 0 and up >= lo:
                flags.append("Upper>=Lower (should be Lower>Upper)")

            if flags:
                distr_rows.append({
                    "Item": item,
                    "Correct": correct,
                    "Distractor": opt,
                    "Overall %": round(ov * 100, 1),
                    "Upper %": round(up * 100, 1),
                    "Lower %": round(lo * 100, 1),
                    "Flags": " | ".join(flags),
                })

distr_df = pd.DataFrame(distr_rows)
flag_counts = distr_df.groupby("Item").size().to_dict() if not distr_df.empty else {}

item_table = pd.DataFrame({
    "Item": item_cols,
    "p (Difficulty)": np.round(p.values, 3),
    "p Category": [difficulty_category(x) for x in p.values],
    "rpbis (Corrected)": np.round(rpbis.values, 3),
    "rpbis Category": [discrim_category(x) for x in rpbis.values],
})

item_table["#DistractorFlags"] = [int(flag_counts.get(it, 0)) for it in item_table["Item"]]
item_table["Decision"] = [
    decision_rule(rpbis.get(it, np.nan), p.get(it, np.nan), int(flag_counts.get(it, 0)))
    for it in item_table["Item"]
]
item_table["Notes"] = [
    item_notes(rpbis.get(it, np.nan), p.get(it, np.nan))
    for it in item_table["Item"]
]

attention_only = st.checkbox(
    "Show only items needing attention (rpbis < .20 OR Decision not KEEP)",
    value=False
)
show_items = item_table.copy()
if attention_only:
    show_items = show_items[
        (show_items["rpbis (Corrected)"] < 0.20) | (show_items["Decision"] != "KEEP")
    ]

st.dataframe(show_items, use_container_width=True)

# Item option bar chart
st.subheader("Item Option Distribution (Bar Chart)")
selected_item = st.selectbox("Select an item", item_cols)

counts = resp[selected_item].value_counts(dropna=False)
counts = pd.Series({opt: int(counts.get(opt, 0)) for opt in options})
fig3, ax3 = plt.subplots()
ax3.bar(counts.index.tolist(), counts.values.tolist())
ax3.set_xlabel("Option")
ax3.set_ylabel("Count")
ax3.set_title(f"Option Distribution - {selected_item} (Correct: {key[selected_item]})")
st.pyplot(fig3)
plt.close(fig3)

st.download_button(
    "Download Item Analysis (CSV)",
    item_table.to_csv(index=False).encode("utf-8"),
    file_name="gtc_item_analysis.csv",
    mime="text/csv",
)

# =========================
# 2) DISTRACTOR ANALYSIS SECOND
# =========================
st.subheader("2) Distractor Analysis")

if upper_idx is None:
    st.warning("Not enough students to compute 27% upper/lower groups reliably.")
else:
    st.write(f"Upper group size = {len(upper_idx)} | Lower group size = {len(lower_idx)} (27% rule)")

    if distr_df.empty:
        st.success("No distractor problems found under current rules.")
    else:
        st.dataframe(distr_df, use_container_width=True)
        st.download_button(
            "Download Distractor Report (CSV)",
            distr_df.to_csv(index=False).encode("utf-8"),
            file_name="gtc_distractor_report.csv",
            mime="text/csv",
        )

# =========================
# Downloads + PDF
# =========================
st.subheader("Downloads")

out01 = pd.concat([df[student_cols], X01[item_cols]], axis=1)

st.download_button(
    "Download 0/1 Data (CSV)",
    out01.to_csv(index=False).encode("utf-8"),
    file_name="gtc_binary_data.csv",
    mime="text/csv",
)

summary_lines = [
    f"N={len(total)} | Mean={float(total.mean()):.2f} | SD={float(total.std(ddof=1)) if len(total)>1 else 0:.2f} | Min={int(total.min())} | Max={int(total.max())}",
    f"KR-20={kr:.3f}" if pd.notna(kr) else "KR-20=N/A",
    f"Unidimensionality: {uni_msg}",
    "Missing treated as incorrect (0) in this MVP.",
]

pdf_bytes = build_pdf_bytes(summary_lines, item_table, distr_df)
st.download_button(
    "Download PDF Report (basic)",
    data=pdf_bytes,
    file_name="gtc_report.pdf",
    mime="application/pdf",
)
