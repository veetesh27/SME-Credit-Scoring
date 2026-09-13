import streamlit as st

from src.backend import assess_sme_final


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SME Credit Assessment System",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("💳 SME Credit Assessment System")

st.markdown(
    """
    ### AI-Powered SME Credit Scoring & Loan Eligibility Prototype

    Enter the business and financial information below to assess
    the SME's credit risk, eligibility, and indicative loan amount.
    """
)

st.info(
    "⚠️ Prototype Notice: This system uses a synthetic dataset and "
    "a project-specific credit scoring model. It is intended for "
    "demonstration and decision-support purposes only, not actual "
    "banking or loan approval."
)


# ============================================================
# BUSINESS INFORMATION
# ============================================================

st.header("🏢 Business Information")

col1, col2, col3 = st.columns(3)

with col1:
    industry_sector = st.selectbox(
        "Industry Sector",
        [
            "Trading & Retail",
            "Manufacturing",
            "Textiles & Apparel",
            "Food & Beverage",
            "IT & ITES Services",
            "Construction & Real Estate",
            "Auto Components",
            "Healthcare & Pharma",
            "Agriculture & Allied",
            "Transportation & Logistics"
        ]
    )

with col2:
    location_type = st.selectbox(
        "Location Type",
        [
            "Urban",
            "Semi-Urban",
            "Rural"
        ]
    )

with col3:
    business_age_years = st.number_input(
        "Business Age (years)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.1
    )


# ============================================================
# FINANCIAL INFORMATION
# ============================================================

st.header("💰 Financial Information")

col1, col2, col3 = st.columns(3)

with col1:
    avg_monthly_revenue = st.number_input(
        "Average Monthly Revenue (INR)",
        min_value=0.0,
        value=1_000_000.0,
        step=10_000.0
    )

with col2:
    avg_monthly_expenses = st.number_input(
        "Average Monthly Expenses (INR)",
        min_value=0.0,
        value=700_000.0,
        step=10_000.0
    )

with col3:
    revenue_growth_rate = st.number_input(
        "Revenue Growth Rate (%)",
        min_value=-100.0,
        max_value=200.0,
        value=7.0,
        step=0.5
    )

col1, col2, col3 = st.columns(3)

with col1:
    cash_flow_volatility = st.number_input(
        "Cash Flow Volatility",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.01
    )

with col2:
    negative_cashflow_months_count = st.number_input(
        "Negative Cash-Flow Months",
        min_value=0,
        max_value=12,
        value=3,
        step=1
    )

with col3:
    avg_monthly_cash_balance = st.number_input(
        "Average Monthly Cash Balance (INR)",
        min_value=0.0,
        value=500_000.0,
        step=10_000.0
    )

col1, col2, col3 = st.columns(3)

with col1:
    min_monthly_balance = st.number_input(
        "Minimum Monthly Balance (INR)",
        min_value=0.0,
        value=300_000.0,
        step=10_000.0
    )

with col2:
    existing_total_liabilities = st.number_input(
        "Existing Total Liabilities (INR)",
        min_value=0.0,
        value=2_500_000.0,
        step=10_000.0
    )

with col3:
    existing_monthly_debt_obligation = st.number_input(
        "Existing Monthly Debt Obligation (INR)",
        min_value=0.0,
        value=80_000.0,
        step=5_000.0
    )

with col1:
    working_capital_cycle_days = st.number_input(
        "Working Capital Cycle (days)",
        min_value=0.0,
        max_value=365.0,
        value=45.0,
        step=1.0
    )


# ============================================================
# BANKING & TRANSACTION INFORMATION
# ============================================================

st.header("🏦 Banking & Transaction Information")

col1, col2, col3 = st.columns(3)

with col1:
    num_transactions_per_month = st.number_input(
        "Transactions per Month",
        min_value=0,
        value=50,
        step=1
    )

with col2:
    credit_transaction_consistency = st.number_input(
        "Credit Transaction Consistency",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01
    )

with col3:
    cash_deposit_frequency = st.number_input(
        "Cash Deposit Frequency",
        min_value=0,
        value=4,
        step=1
    )

col1, col2 = st.columns(2)

with col1:
    bounced_payment_count = st.number_input(
        "Bounced Payment Count",
        min_value=0,
        value=1,
        step=1
    )

with col2:
    recurring_revenue_ratio = st.number_input(
        "Recurring Revenue Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01
    )


# ============================================================
# GST INFORMATION
# ============================================================

st.header("🧾 GST Information")

col1, col2, col3 = st.columns(3)

with col1:
    gst_filing_regularity_score = st.number_input(
        "GST Filing Regularity Score",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01
    )

with col2:
    gst_late_filings_count = st.number_input(
        "GST Late Filings Count",
        min_value=0,
        value=1,
        step=1
    )

with col3:
    gst_turnover_consistency = st.number_input(
        "GST Turnover Consistency",
        min_value=0.0,
        max_value=1.0,
        value=0.72,
        step=0.01
    )


# ============================================================
# DIGITAL PRESENCE
# ============================================================

st.header("🌐 Digital Presence")

col1, col2, col3 = st.columns(3)

with col1:
    has_website = st.checkbox(
        "Has Website",
        value=True
    )

with col2:
    has_google_listing = st.checkbox(
        "Has Google Listing",
        value=True
    )

with col3:
    has_social_media = st.checkbox(
        "Has Social Media",
        value=False
    )


# ============================================================
# ONLINE REPUTATION
# ============================================================

st.header("⭐ Online Reputation")

col1, col2 = st.columns(2)

with col1:
    online_review_count = st.number_input(
        "Online Review Count",
        min_value=0,
        value=25,
        step=1
    )

with col2:
    online_review_rating = st.number_input(
        "Online Review Rating",
        min_value=0.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )


# ============================================================
# ASSESSMENT BUTTON
# ============================================================

st.divider()

if st.button(
    "🔍 Assess SME Credit",
    type="primary",
    use_container_width=True
):

    # Convert checkbox values to the 0/1 format
    # expected by the backend.
    applicant = {
        "industry_sector": industry_sector,
        "location_type": location_type,
        "business_age_years": business_age_years,

        "avg_monthly_revenue": avg_monthly_revenue,
        "revenue_growth_rate": revenue_growth_rate,
        "avg_monthly_expenses": avg_monthly_expenses,
        "cash_flow_volatility": cash_flow_volatility,
        "negative_cashflow_months_count": negative_cashflow_months_count,
        "avg_monthly_cash_balance": avg_monthly_cash_balance,
        "min_monthly_balance": min_monthly_balance,
        "existing_total_liabilities": existing_total_liabilities,
        "existing_monthly_debt_obligation": existing_monthly_debt_obligation,
        "working_capital_cycle_days": working_capital_cycle_days,

        "num_transactions_per_month": num_transactions_per_month,
        "credit_transaction_consistency": credit_transaction_consistency,
        "cash_deposit_frequency": cash_deposit_frequency,
        "bounced_payment_count": bounced_payment_count,
        "recurring_revenue_ratio": recurring_revenue_ratio,

        "gst_filing_regularity_score": gst_filing_regularity_score,
        "gst_late_filings_count": gst_late_filings_count,
        "gst_turnover_consistency": gst_turnover_consistency,

        "has_website": int(has_website),
        "has_google_listing": int(has_google_listing),
        "has_social_media": int(has_social_media),

        "online_review_count": online_review_count,
        "online_review_rating": online_review_rating
    }

    try:

        with st.spinner("Assessing SME credit risk..."):

            result = assess_sme_final(applicant)

        st.session_state["assessment_result"] = result

        st.success("Assessment completed successfully!")

    except Exception as e:

        st.error("An error occurred while assessing the applicant.")

        st.exception(e)


# ============================================================
# RESULT DASHBOARD
# ============================================================

if "assessment_result" in st.session_state:

    result = st.session_state["assessment_result"]

    st.divider()

    st.header("📊 Credit Assessment Result")

    # Extract results
    credit_score = result["credit_score"]
    default_risk = result["default_probability"]
    risk_category = result["risk_category"]
    eligibility_status = result["eligibility_status"]

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "💳 Credit Score",
            f"{credit_score:.0f} / 900"
        )

    with col2:
        st.metric(
            "📉 Model Default Risk",
            f"{default_risk * 100:.2f}%"
        )

    with col3:
        st.metric(
            "⚠️ Risk Category",
            risk_category
        )

    # --------------------------------------------------------
    # SCORE INTERPRETATION
    # --------------------------------------------------------

    st.subheader("📌 Score Interpretation")

    if risk_category == "Excellent":

        st.success(
            "Excellent credit profile. The applicant demonstrates "
            "strong financial and business characteristics."
        )

    elif risk_category == "Good":

        st.success(
            "Good credit profile. The applicant generally demonstrates "
            "healthy creditworthiness with manageable risk."
        )

    elif risk_category == "Moderate":

        st.warning(
            "Moderate credit risk. The applicant may be eligible, "
            "but enhanced due diligence is recommended."
        )

    elif risk_category == "High Risk":

        st.warning(
            "High credit risk. The applicant requires careful "
            "manual review before lending."
        )

    else:

        st.error(
            "Very high credit risk. The applicant does not meet "
            "the prototype's lending criteria."
        )

    # --------------------------------------------------------
    # CREDIT SCORE PROGRESS
    # --------------------------------------------------------

    st.progress(
        min(max(credit_score / 900, 0.0), 1.0)
    )

    st.caption(
        "Project-specific credit score ranging from 300 to 900. "
        "This is not an official CIBIL or bureau score."
    )

    # --------------------------------------------------------
    # ELIGIBILITY
    # --------------------------------------------------------

    st.subheader("🏦 Loan Eligibility")

    if eligibility_status.startswith("Eligible"):

        st.success(
            f"✅ {eligibility_status}"
        )

    else:

        st.error(
            f"❌ {eligibility_status}"
        )

    # --------------------------------------------------------
    # FAILED BUSINESS RULES
    # --------------------------------------------------------

    failed_rules = result.get("failed_rules", [])

    if failed_rules:

        st.markdown("### 🚨 Business Rule Warnings")

        for rule in failed_rules:

            st.warning(rule)

    else:

        st.success(
            "No business-rule violations were detected."
        )


    # ========================================================
    # LOAN RECOMMENDATION
    # ========================================================

    st.subheader("💰 Indicative Loan Recommendation")

    loan = result["loan_recommendation"]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Recommended Loan Amount",
            f"₹{loan['recommended_loan_amount']:,.0f}"
        )

    with col2:

        st.metric(
            "Interest Rate",
            f"{loan['assumed_interest_rate'] * 100:.0f}%"
        )

    with col3:

        st.metric(
            "Tenure",
            f"{loan['assumed_tenure_months']} months"
        )


# ========================================================
# SHAP EXPLANATION
# ========================================================

if "assessment_result" in st.session_state:

    result = st.session_state["assessment_result"]

    st.subheader("🔎 Why This Result?")

    shap_df = result["shap_explanation"].copy()

    FEATURE_NAMES = {
        "business_age_years": "Business Age",
        "industry_risk_numeric": "Industry Risk Level",
        "industry_growth_rate": "Industry Growth Rate",
        "avg_monthly_revenue": "Average Monthly Revenue",
        "revenue_growth_rate": "Revenue Growth Rate",
        "profit_margin": "Profit Margin",
        "cash_flow_volatility": "Cash-Flow Volatility",
        "negative_cashflow_months_count": "Negative Cash-Flow Months",
        "min_monthly_balance": "Minimum Monthly Balance",
        "existing_total_liabilities": "Existing Total Liabilities",
        "existing_monthly_debt_obligation": "Existing Monthly Debt Obligation",
        "debt_service_ratio": "Debt Service Ratio",
        "working_capital_cycle_days": "Working Capital Cycle",
        "num_transactions_per_month": "Monthly Transactions",
        "credit_transaction_consistency": "Transaction Consistency",
        "cash_deposit_frequency": "Cash Deposit Frequency",
        "bounced_payment_count": "Bounced Payments",
        "recurring_revenue_ratio": "Recurring Revenue Ratio",
        "gst_filing_regularity_score": "GST Filing Regularity",
        "gst_late_filings_count": "GST Late Filings",
        "gst_turnover_consistency": "GST Turnover Consistency",
        "has_website": "Website Presence",
        "has_google_listing": "Google Listing",
        "has_social_media": "Social Media Presence",
        "online_review_count": "Online Review Count",
        "online_review_rating": "Online Review Rating",
        "local_unemployment_rate": "Local Unemployment Rate",
        "local_inflation_rate": "Local Inflation Rate",
        "local_demand_index": "Local Demand Index",
        "market_competition_score": "Market Competition",
        "regional_business_growth_rate": "Regional Business Growth",
        "cash_buffer_months": "Cash Buffer",
        "net_monthly_cashflow": "Net Monthly Cash Flow",
        "free_cash_flow_ratio": "Free Cash Flow Ratio",
        "liabilities_to_annual_revenue": "Liabilities / Annual Revenue"
    }

    shap_df["display_name"] = shap_df["feature"].map(
        lambda x: FEATURE_NAMES.get(x, x)
    )

    risk_increasing = shap_df[
        shap_df["shap_value"] > 0
    ].sort_values("impact", ascending=False)

    risk_reducing = shap_df[
        shap_df["shap_value"] < 0
    ].sort_values("impact", ascending=False)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🔴 Factors Increasing Risk")

        if len(risk_increasing) == 0:

            st.success(
                "No major risk-increasing factors identified."
            )

        else:

            for _, row in risk_increasing.iterrows():

                st.markdown(
                    f"""
                    **{row['display_name']}**

                    Model impact: **{row['impact']:.3f}**
                    """
                )

                st.progress(
                    min(float(row["impact"]), 1.0)
                )

    with col2:

        st.markdown("### 🟢 Factors Reducing Risk")

        if len(risk_reducing) == 0:

            st.info(
                "No major risk-reducing factors identified."
            )

        else:

            for _, row in risk_reducing.iterrows():

                st.markdown(
                    f"""
                    **{row['display_name']}**

                    Model impact: **{row['impact']:.3f}**
                    """
                )

                st.progress(
                    min(float(row["impact"]), 1.0)
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SME Credit Scoring and Loan Eligibility System — "
    "Prototype | Synthetic Data | Decision-Support Only"
)