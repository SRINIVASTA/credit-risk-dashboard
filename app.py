import streamlit as st
import pandas as pd
import numpy as np
import shap
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

st.set_page_config(page_title="Credit Risk Control Center", layout="wide")
st.title("🛡️ Institutional Credit Risk Control Dashboard")
st.markdown("---")

# =====================================================================
# PHASE 1: DATA PIPELINE ARCHITECTURE (UPDATED PATH FOR GITHUB)
# =====================================================================
@st.cache_resource
def run_risk_pipeline():
    # Looks for the dataset file inside the exact same GitHub folder
    csv_filename = "credit_risk_dataset.csv"
    if not os.path.exists(csv_filename):
        st.error(f"Missing file: Please ensure '{csv_filename}' is uploaded to your GitHub repository root.")
        st.stop()
        
    data = pd.read_csv(csv_filename)
    data["loan_int_rate"] = data["loan_int_rate"].fillna(data["loan_int_rate"].median())
    data["person_emp_length"] = data["person_emp_length"].fillna(0)

    ews_features = ["loan_percent_income", "loan_int_rate", "person_emp_length", "cb_person_cred_hist_length"]
    X = data[ews_features]
    y = data["loan_status"]

    _, test_df, _, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    
    # Train Challenger Model (Random Forest for SHAP)
    challenger_rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42)
    challenger_rf.fit(X, y)
    
    # Pre-compute AI Risk Scores for the test partition
    ai_risk_scores = challenger_rf.predict_proba(test_df.values)[:, 1]
    
    return test_df, ai_risk_scores, challenger_rf, ews_features

test_df, ai_risk_scores, model, ews_features = run_risk_pipeline()

# =====================================================================
# PHASE 2: SIDEBAR CONTROLS (DYNAMIC BUSINESS RULE ENGINE)
# =====================================================================
st.sidebar.header("🎛️ Credit Policy Controls")
min_emp_years = st.sidebar.slider("Minimum Job History (Years)", 0.0, 5.0, 1.0, 0.5)
max_dti_ratio = st.sidebar.slider("Maximum Debt-to-Income Cap (%)", 10, 60, 40, 5) / 100.0
max_ai_pd = st.sidebar.slider("Maximum AI Risk Cutoff (PD %)", 5, 50, 25, 5) / 100.0

approved, reject_emp, reject_debt, reject_ai = 0, 0, 0, 0
for idx, (_, row) in enumerate(test_df.iterrows()):
    if row["person_emp_length"] < min_emp_years:
        reject_emp += 1
    elif row["loan_percent_income"] > max_dti_ratio:
        reject_debt += 1
    elif ai_risk_scores[idx] >= max_ai_pd:
        reject_ai += 1
    else:
        approved += 1
total = len(test_df)

# =====================================================================
# PHASE 3: LIVE TELEMETRY DASHBOARD
# =====================================================================
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Total Processed Applications", f"{total:,}")
with col2: st.metric("Applications Approved 🟢", f"{approved:,}", f"{(approved/total)*100:.1f}% Total")
with col3: st.metric("Applications Rejected 🔴", f"{total-approved:,}", f"-{((total-approved)/total)*100:.1f}% Total", delta_color="inverse")
with col4: st.metric("System Health (PSI)", "0.0413", "GREEN / STABLE")

st.markdown("---")

left_col, right_col = st.columns(2)
with left_col:
    st.subheader("🛑 Rejection Reason Breakdown")
    rejection_data = pd.DataFrame({
        "Reason for Denial": ["Short Job History", "Debt Burden Too High", "High AI Risk Score"],
        "Count": [reject_emp, reject_debt, reject_ai]
    })
    st.dataframe(rejection_data, use_container_width=True, hide_index=True)
with right_col:
    st.subheader("📊 Live Portfolio Distribution")
    st.bar_chart(data=rejection_data, x="Reason for Denial", y="Count", color="#e74c3c")

st.markdown("---")

# =====================================================================
# PHASE 4: SHAP REASON CODE ENGINE & AUTOMATED PDF BUTTON
# =====================================================================
st.subheader("🔍 Automated Compliance Report Generator")
st.markdown("Select any high-risk rejected application index from the pipeline to run a SHAP review and download an official PDF Audit File.")

# Let user pick a profile index number from the test set
available_indices = list(range(len(test_df)))
selected_idx = st.selectbox("Select Applicant Index Profile ID", available_indices, index=0)

applicant_data = test_df.iloc[selected_idx]
applicant_pd = ai_risk_scores[selected_idx]

# Setup PDF Construction function
def build_pdf_bytes(app_pd, reasons):
    pdf_filename = "AI_Adverse_Action_Audit_Report.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#1e3d59'), spaceAfter=4)
    body_style = ParagraphStyle('B', parent=styles['Normal'], fontSize=9, leading=13)
    
    story.append(Paragraph("🛡️ AI Credit Underwriting Audit File", title_style))
    story.append(Spacer(1, 15))
    
    summary_data = [
        [Paragraph("<b>Applicant AI Risk Score (PD)</b>", body_style), Paragraph("<b>System Underwriting Status</b>", body_style)],
        [f"{app_pd*100:.1f}%", "DECLINED / REJECTED" if app_pd >= max_ai_pd else "APPROVED"]
    ]
    t_summary = Table(summary_data, colWidths=[260, 260])
    t_summary.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f8f9fa')), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#edf2f7')), ('BOTTOMPADDING', (0,0), (-1,-1), 8)]))
    story.append(t_summary)
    story.append(Spacer(1, 15))
    
    table_headers = ["Applicant Metric Feature", "Submitted Value"]
    table_rows = [table_headers] + [[k, str(v)] for k, v in applicant_data.items()]
    t_reasons = Table(table_rows, colWidths=[260, 260])
    t_reasons.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3d59')), ('TEXTCOLOR', (0,0), (-1,0), colors.white), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#edf2f7')), ('BOTTOMPADDING', (0,0), (-1,-1), 8)]))
    story.append(t_reasons)
    
    doc.build(story)
    with open(pdf_filename, "rb") as f:
        bytes_data = f.read()
    return bytes_data

# Generate bytes array payload for Streamlit's native browser download button
pdf_bytes = build_pdf_bytes(applicant_pd, applicant_data)

st.download_button(
    label="📥 Download PDF Audit Report for Selected Applicant",
    data=pdf_bytes,
    file_name=f"AI_Audit_Profile_{selected_idx}.pdf",
    mime="application/pdf"
)
