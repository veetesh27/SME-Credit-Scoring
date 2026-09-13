# ============================================================
# SME CREDIT SCORING SYSTEM — LOCAL BACKEND
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sme_credit_xgb.json"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "scoring_configuration.json"
)

FEATURE_COLUMNS_PATH = os.path.join(
    MODEL_DIR,
    "model_feature_columns.json"
)

INDUSTRY_TABLE_PATH = os.path.join(
    MODEL_DIR,
    "industry_table.json"
)

LOCATION_TABLE_PATH = os.path.join(
    MODEL_DIR,
    "location_table.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

loaded_model = xgb.XGBClassifier()

loaded_model.load_model(
    MODEL_PATH
)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

with open(CONFIG_PATH, "r") as f:
    scoring_config = json.load(f)


with open(FEATURE_COLUMNS_PATH, "r") as f:
    MODEL_FEATURE_COLUMNS = json.load(f)


with open(INDUSTRY_TABLE_PATH, "r") as f:
    INDUSTRY_TABLE = json.load(f)


with open(LOCATION_TABLE_PATH, "r") as f:
    LOCATION_TABLE = json.load(f)


# ============================================================
# RISK MAPPING
# ============================================================

RISK_NUM = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}


# ============================================================
# SCORE CONFIGURATION
# ============================================================

SCORE_MIN = scoring_config[
    "score_range"
]["min"]

SCORE_MAX = scoring_config[
    "score_range"
]["max"]


P1 = scoring_config[
    "score_formula"
]["P1"]

P99 = scoring_config[
    "score_formula"
]["P99"]

SCORE_FACTOR = scoring_config[
    "score_formula"
]["SCORE_FACTOR"]

SCORE_OFFSET = scoring_config[
    "score_formula"
]["SCORE_OFFSET"]


# ============================================================
# ELIGIBILITY CONFIGURATION
# ============================================================

eligibility_config = scoring_config[
    "eligibility_rules"
]

MAX_DEBT_SERVICE_RATIO = eligibility_config[
    "MAX_DEBT_SERVICE_RATIO"
]

MIN_BUSINESS_AGE_YEARS = eligibility_config[
    "MIN_BUSINESS_AGE_YEARS"
]

MIN_GST_FILING_REGULARITY = eligibility_config[
    "MIN_GST_FILING_REGULARITY"
]

MIN_NET_MONTHLY_CASHFLOW = eligibility_config[
    "MIN_NET_MONTHLY_CASHFLOW"
]


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    loaded_model
)


# ============================================================
# RAW APPLICANT → MODEL FEATURES
# ============================================================

def prepare_applicant_features(applicant):

    applicant = applicant.copy()

    required_fields = [
        'industry_sector',
        'location_type',
        'business_age_years',
        'avg_monthly_revenue',
        'revenue_growth_rate',
        'avg_monthly_expenses',
        'cash_flow_volatility',
        'negative_cashflow_months_count',
        'avg_monthly_cash_balance',
        'min_monthly_balance',
        'existing_total_liabilities',
        'existing_monthly_debt_obligation',
        'working_capital_cycle_days',
        'num_transactions_per_month',
        'credit_transaction_consistency',
        'cash_deposit_frequency',
        'bounced_payment_count',
        'recurring_revenue_ratio',
        'gst_filing_regularity_score',
        'gst_late_filings_count',
        'gst_turnover_consistency',
        'has_website',
        'has_google_listing',
        'has_social_media',
        'online_review_count',
        'online_review_rating'
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in applicant
    ]

    if missing_fields:
        raise ValueError(
            f"Missing required applicant fields: {missing_fields}"
        )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if applicant['avg_monthly_revenue'] <= 0:
        raise ValueError(
            "Average monthly revenue must be greater than 0."
        )

    if applicant['avg_monthly_expenses'] < 0:
        raise ValueError(
            "Average monthly expenses cannot be negative."
        )

    if applicant['business_age_years'] < 0:
        raise ValueError(
            "Business age cannot be negative."
        )

    # --------------------------------------------------------
    # Industry information
    # --------------------------------------------------------

    industry = applicant[
        'industry_sector'
    ]

    if industry not in INDUSTRY_TABLE:
        raise ValueError(
            f"Unknown industry sector: {industry}"
        )

    industry_info = INDUSTRY_TABLE[
        industry
    ]

    industry_growth_rate = (
        industry_info['growth']
    )

    industry_risk_level = (
        industry_info['risk']
    )

    industry_risk_numeric = (
        RISK_NUM[industry_risk_level]
    )

    # --------------------------------------------------------
    # Location information
    # --------------------------------------------------------

    location = applicant[
        'location_type'
    ]

    if location not in LOCATION_TABLE:
        raise ValueError(
            f"Unknown location type: {location}"
        )

    location_info = LOCATION_TABLE[
        location
    ]

    local_unemployment_rate = (
        location_info['unemployment']
    )

    local_inflation_rate = (
        location_info['inflation']
    )

    local_demand_index = (
        location_info['demand']
    )

    market_competition_score = (
        location_info['competition']
    )

    regional_business_growth_rate = (
        location_info['growth']
    )

    # --------------------------------------------------------
    # Financial feature engineering
    # --------------------------------------------------------

    revenue = applicant[
        'avg_monthly_revenue'
    ]

    expenses = applicant[
        'avg_monthly_expenses'
    ]

    debt_payment = applicant[
        'existing_monthly_debt_obligation'
    ]

    profit_margin = (
        (revenue - expenses)
        / revenue
    ) * 100

    net_monthly_cashflow = (
        revenue
        - expenses
        - debt_payment
    )

    free_cash_flow_ratio = (
        net_monthly_cashflow
        / revenue
    )

    debt_service_ratio = (
        debt_payment
        / revenue
    )

    cash_buffer_months = (
        applicant[
            'avg_monthly_cash_balance'
        ] / expenses
        if expenses > 0
        else 0
    )

    liabilities_to_annual_revenue = (
        applicant[
            'existing_total_liabilities'
        ]
        / (revenue * 12)
    )

    # --------------------------------------------------------
    # Build feature dictionary
    # --------------------------------------------------------

    data = {

        'business_age_years':
            applicant['business_age_years'],

        'industry_risk_numeric':
            industry_risk_numeric,

        'industry_growth_rate':
            industry_growth_rate,

        'avg_monthly_revenue':
            revenue,

        'revenue_growth_rate':
            applicant['revenue_growth_rate'],

        'profit_margin':
            profit_margin,

        'cash_flow_volatility':
            applicant['cash_flow_volatility'],

        'negative_cashflow_months_count':
            applicant['negative_cashflow_months_count'],

        'min_monthly_balance':
            applicant['min_monthly_balance'],

        'existing_total_liabilities':
            applicant['existing_total_liabilities'],

        'existing_monthly_debt_obligation':
            debt_payment,

        'debt_service_ratio':
            debt_service_ratio,

        'working_capital_cycle_days':
            applicant[
                'working_capital_cycle_days'
            ],

        'num_transactions_per_month':
            applicant[
                'num_transactions_per_month'
            ],

        'credit_transaction_consistency':
            applicant[
                'credit_transaction_consistency'
            ],

        'cash_deposit_frequency':
            applicant[
                'cash_deposit_frequency'
            ],

        'bounced_payment_count':
            applicant[
                'bounced_payment_count'
            ],

        'recurring_revenue_ratio':
            applicant[
                'recurring_revenue_ratio'
            ],

        'gst_filing_regularity_score':
            applicant[
                'gst_filing_regularity_score'
            ],

        'gst_late_filings_count':
            applicant[
                'gst_late_filings_count'
            ],

        'gst_turnover_consistency':
            applicant[
                'gst_turnover_consistency'
            ],

        'has_website':
            applicant['has_website'],

        'has_google_listing':
            applicant['has_google_listing'],

        'has_social_media':
            applicant['has_social_media'],

        'online_review_count':
            applicant['online_review_count'],

        'online_review_rating':
            applicant['online_review_rating'],

        'local_unemployment_rate':
            local_unemployment_rate,

        'local_inflation_rate':
            local_inflation_rate,

        'local_demand_index':
            local_demand_index,

        'market_competition_score':
            market_competition_score,

        'regional_business_growth_rate':
            regional_business_growth_rate,

        'cash_buffer_months':
            cash_buffer_months,

        'net_monthly_cashflow':
            net_monthly_cashflow,

        'free_cash_flow_ratio':
            free_cash_flow_ratio,

        'liabilities_to_annual_revenue':
            liabilities_to_annual_revenue
    }

    row = pd.DataFrame([data])

    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    row['industry_sector'] = industry

    row['location_type'] = location

    row = pd.get_dummies(
        row,
        columns=[
            'industry_sector',
            'location_type'
        ],
        drop_first=True
    )

    # --------------------------------------------------------
    # Add missing model columns
    # --------------------------------------------------------

    for column in MODEL_FEATURE_COLUMNS:

        if column not in row.columns:
            row[column] = 0

    # --------------------------------------------------------
    # Exact model feature order
    # --------------------------------------------------------

    row = row[
        MODEL_FEATURE_COLUMNS
    ]

    return row


# ============================================================
# CREDIT SCORE
# ============================================================

def logit(p):

    p = np.clip(
        p,
        1e-6,
        1 - 1e-6
    )

    return np.log(
        p / (1 - p)
    )


def probability_to_credit_score(
    probability
):

    probability = np.asarray(
        probability,
        dtype=float
    )

    raw_score = (
        SCORE_OFFSET
        + SCORE_FACTOR
        * logit(probability)
    )

    return np.clip(
        raw_score,
        SCORE_MIN,
        SCORE_MAX
    )


# ============================================================
# RISK CATEGORY
# ============================================================

def risk_category(score):

    if score >= 750:
        return 'Excellent'

    elif score >= 650:
        return 'Good'

    elif score >= 550:
        return 'Moderate'

    elif score >= 450:
        return 'High Risk'

    else:
        return 'Very High Risk'


# ============================================================
# ELIGIBILITY ENGINE
# ============================================================

def loan_eligibility_decision(
    credit_score_value,
    business_age_years,
    debt_service_ratio,
    gst_filing_regularity_score,
    net_monthly_cashflow
):

    category = risk_category(
        credit_score_value
    )

    failed_rules = []

    if business_age_years < MIN_BUSINESS_AGE_YEARS:

        failed_rules.append(
            f"Business age "
            f"({business_age_years:.1f} yrs) "
            f"is below the "
            f"{MIN_BUSINESS_AGE_YEARS}-yr minimum"
        )

    if debt_service_ratio > MAX_DEBT_SERVICE_RATIO:

        failed_rules.append(
            f"Debt-service ratio "
            f"({debt_service_ratio:.2f}) "
            f"exceeds the "
            f"{MAX_DEBT_SERVICE_RATIO} maximum"
        )

    if (
        gst_filing_regularity_score
        < MIN_GST_FILING_REGULARITY
    ):

        failed_rules.append(
            f"GST filing regularity "
            f"({gst_filing_regularity_score:.2f}) "
            f"is below the "
            f"{MIN_GST_FILING_REGULARITY} minimum"
        )

    if (
        net_monthly_cashflow
        < MIN_NET_MONTHLY_CASHFLOW
    ):

        failed_rules.append(
            f"Net monthly cash flow "
            f"(INR {net_monthly_cashflow:,.0f}) "
            f"is negative"
        )

    if failed_rules:

        eligibility_status = (
            "Not Eligible "
            "(Business Rule Failure)"
        )

    elif category == 'Excellent':

        eligibility_status = (
            "Eligible - Preferred Terms"
        )

    elif category == 'Good':

        eligibility_status = (
            "Eligible - Standard Terms"
        )

    elif category == 'Moderate':

        eligibility_status = (
            "Eligible - Enhanced Due Diligence"
        )

    elif category == 'High Risk':

        eligibility_status = (
            "Not Eligible - "
            "Refer for Manual Review"
        )

    else:

        eligibility_status = (
            "Not Eligible"
        )

    return {
        'risk_category': category,
        'eligibility_status':
            eligibility_status,
        'failed_rules':
            failed_rules
    }


# ============================================================
# LOAN OPTIMIZER
# ============================================================

LOAN_ANNUAL_INTEREST_RATE = 0.12
LOAN_TENURE_MONTHS = 36

RISK_ADJUSTMENTS = {

    'Excellent': 1.00,
    'Good': 0.85,
    'Moderate': 0.65,
    'High Risk': 0.40,
    'Very High Risk': 0.00
}


def calculate_emi(
    principal,
    annual_rate,
    months
):

    monthly_rate = (
        annual_rate / 12
    )

    if monthly_rate == 0:

        return principal / months

    emi = (
        principal
        * monthly_rate
        * (1 + monthly_rate) ** months
        /
        (
            (1 + monthly_rate) ** months
            - 1
        )
    )

    return emi


def calculate_max_loan_from_emi(
    available_monthly_payment,
    annual_rate=LOAN_ANNUAL_INTEREST_RATE,
    months=LOAN_TENURE_MONTHS
):

    if available_monthly_payment <= 0:
        return 0

    monthly_rate = (
        annual_rate / 12
    )

    if monthly_rate == 0:

        return (
            available_monthly_payment
            * months
        )

    loan_amount = (
        available_monthly_payment
        * (
            (1 + monthly_rate)
            ** months - 1
        )
        /
        (
            monthly_rate
            * (1 + monthly_rate)
            ** months
        )
    )

    return loan_amount


def recommend_loan_amount(
    applicant,
    credit_score,
    eligibility_status
):

    revenue = applicant[
        'avg_monthly_revenue'
    ]

    existing_debt_payment = applicant[
        'existing_monthly_debt_obligation'
    ]

    # Maximum total debt payment
    max_total_debt_service = (
        revenue
        * MAX_DEBT_SERVICE_RATIO
    )

    # Available payment for new loan
    available_new_payment = (
        max_total_debt_service
        - existing_debt_payment
    )

    available_new_payment = max(
        available_new_payment,
        0
    )

    # Convert EMI capacity to loan amount
    serviceability_amount = (
        calculate_max_loan_from_emi(
            available_new_payment
        )
    )

    category = risk_category(
        credit_score
    )

    adjustment = RISK_ADJUSTMENTS[
        category
    ]

    recommended_amount = (
        serviceability_amount
        * adjustment
    )

    # Hard eligibility failure
    if eligibility_status.startswith(
        "Not Eligible"
    ):

        recommended_amount = 0

    return {

        'maximum_monthly_debt_service':
            round(
                max_total_debt_service,
                2
            ),

        'available_new_debt_service':
            round(
                available_new_payment,
                2
            ),

        'serviceability_loan_amount':
            round(
                serviceability_amount,
                2
            ),

        'risk_adjustment':
            adjustment,

        'recommended_loan_amount':
            round(
                recommended_amount,
                2
            ),

        'assumed_interest_rate':
            LOAN_ANNUAL_INTEREST_RATE,

        'assumed_tenure_months':
            LOAN_TENURE_MONTHS
    }


# ============================================================
# SHAP EXPLANATION
# ============================================================

def explain_applicant(
    applicant,
    top_n=8
):

    feature_vector = (
        prepare_applicant_features(
            applicant
        )
    )

    values = explainer.shap_values(
        feature_vector
    )

    values = np.asarray(
        values
    )

    if values.ndim == 2:
        values = values[0]

    explanation = pd.DataFrame({

        'feature':
            feature_vector.columns,

        'feature_value':
            feature_vector.iloc[0].values,

        'shap_value':
            values
    })

    explanation['direction'] = np.where(

        explanation['shap_value'] > 0,

        'Increases default risk',

        'Reduces default risk'
    )

    explanation['impact'] = (
        explanation['shap_value'].abs()
    )

    explanation = explanation.sort_values(
        'impact',
        ascending=False
    )

    return explanation.head(
        top_n
    )


# ============================================================
# HUMAN-READABLE FEATURE NAMES
# ============================================================

FEATURE_DISPLAY_NAMES = {

    'business_age_years':
        'Business Age',

    'industry_risk_numeric':
        'Industry Risk',

    'industry_growth_rate':
        'Industry Growth',

    'avg_monthly_revenue':
        'Monthly Revenue',

    'revenue_growth_rate':
        'Revenue Growth',

    'profit_margin':
        'Profit Margin',

    'cash_flow_volatility':
        'Cash Flow Volatility',

    'negative_cashflow_months_count':
        'Negative Cash-Flow Months',

    'min_monthly_balance':
        'Minimum Bank Balance',

    'existing_total_liabilities':
        'Existing Liabilities',

    'existing_monthly_debt_obligation':
        'Existing Monthly Debt',

    'debt_service_ratio':
        'Debt-Service Ratio',

    'working_capital_cycle_days':
        'Working Capital Cycle',

    'num_transactions_per_month':
        'Monthly Transactions',

    'credit_transaction_consistency':
        'Transaction Consistency',

    'cash_deposit_frequency':
        'Cash Deposit Frequency',

    'bounced_payment_count':
        'Bounced Payments',

    'recurring_revenue_ratio':
        'Recurring Revenue',

    'gst_filing_regularity_score':
        'GST Filing Regularity',

    'gst_late_filings_count':
        'GST Late Filings',

    'gst_turnover_consistency':
        'GST Turnover Consistency',

    'has_website':
        'Website Presence',

    'has_google_listing':
        'Google Listing',

    'has_social_media':
        'Social Media Presence',

    'online_review_count':
        'Online Reviews',

    'online_review_rating':
        'Online Review Rating',

    'local_unemployment_rate':
        'Local Unemployment',

    'local_inflation_rate':
        'Local Inflation',

    'local_demand_index':
        'Local Demand',

    'market_competition_score':
        'Market Competition',

    'regional_business_growth_rate':
        'Regional Business Growth',

    'cash_buffer_months':
        'Cash Buffer',

    'net_monthly_cashflow':
        'Net Monthly Cash Flow',

    'free_cash_flow_ratio':
        'Free Cash Flow Ratio',

    'liabilities_to_annual_revenue':
        'Liabilities / Annual Revenue'
}


def humanize_feature_name(
    feature
):

    if feature in FEATURE_DISPLAY_NAMES:

        return FEATURE_DISPLAY_NAMES[
            feature
        ]

    if feature.startswith(
        'industry_sector_'
    ):

        return (
            "Industry: "
            + feature.replace(
                'industry_sector_',
                ''
            )
        )

    if feature.startswith(
        'location_type_'
    ):

        return (
            "Location: "
            + feature.replace(
                'location_type_',
                ''
            )
        )

    return feature.replace(
        '_',
        ' '
    ).title()


# ============================================================
# FINAL END-TO-END ASSESSMENT
# ============================================================

def assess_sme(
    applicant
):

    # 1. Feature preparation
    feature_vector = (
        prepare_applicant_features(
            applicant
        )
    )

    # 2. Default probability
    probability = float(
        loaded_model.predict_proba(
            feature_vector
        )[0, 1]
    )

    # 3. Credit score
    credit_score = float(
        probability_to_credit_score(
            probability
        )
    )

    # 4. Risk category
    category = risk_category(
        credit_score
    )

    # 5. Eligibility
    eligibility = (
        loan_eligibility_decision(
            credit_score_value=credit_score,

            business_age_years=applicant[
                'business_age_years'
            ],

            debt_service_ratio=
                feature_vector[
                    'debt_service_ratio'
                ].iloc[0],

            gst_filing_regularity_score=
                applicant[
                    'gst_filing_regularity_score'
                ],

            net_monthly_cashflow=
                feature_vector[
                    'net_monthly_cashflow'
                ].iloc[0]
        )
    )

    # 6. Loan recommendation
    loan = recommend_loan_amount(
        applicant,
        credit_score,
        eligibility[
            'eligibility_status'
        ]
    )

    # 7. SHAP explanation
    shap_explanation = (
        explain_applicant(
            applicant,
            top_n=8
        )
    )

    return {

        'default_probability':
            probability,

        'credit_score':
            round(
                credit_score,
                2
            ),

        'risk_category':
            category,

        'eligibility_status':
            eligibility[
                'eligibility_status'
            ],

        'failed_rules':
            eligibility[
                'failed_rules'
            ],

        'loan_recommendation':
            loan,

        'shap_explanation':
            shap_explanation,

        'feature_vector':
            feature_vector
    }

# ============================================================
# FINAL SME ASSESSMENT ENGINE
# ============================================================

def assess_sme_final(applicant):

    # Prepare the applicant's 46-feature model input
    feature_vector = prepare_applicant_features(applicant)

    # Predict default risk using the saved XGBoost model
    probability = float(
        loaded_model.predict_proba(feature_vector)[0, 1]
    )

    # Convert model output into project-specific credit score
    credit_score = float(
        probability_to_credit_score(probability)
    )

    # Determine risk category
    category = risk_category(credit_score)

    # Apply hard eligibility rules
    eligibility = loan_eligibility_decision(
        credit_score_value=credit_score,
        business_age_years=applicant['business_age_years'],
        debt_service_ratio=feature_vector['debt_service_ratio'].iloc[0],
        gst_filing_regularity_score=applicant['gst_filing_regularity_score'],
        net_monthly_cashflow=feature_vector['net_monthly_cashflow'].iloc[0]
    )

    # Calculate recommended loan amount
    loan = recommend_loan_amount(
        applicant,
        credit_score,
        eligibility['eligibility_status']
    )

    # Generate SHAP explanation
    shap_explanation = explain_applicant(
        applicant,
        top_n=8
    )

    return {
        'default_probability': probability,
        'credit_score': round(credit_score, 2),
        'risk_category': category,
        'eligibility_status': eligibility['eligibility_status'],
        'failed_rules': eligibility['failed_rules'],
        'loan_recommendation': loan,
        'shap_explanation': shap_explanation,
        'feature_vector': feature_vector
    }