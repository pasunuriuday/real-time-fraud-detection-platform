import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
from psycopg.rows import dict_row
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.storage.alert_manager import (
    assign_alert,
    get_alerts,
    get_alert_statistics,
    update_alert,
    update_analyst_notes,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudShield | Fraud Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# AUTO REFRESH
# ============================================================

st_autorefresh(
    interval=5000,
    key="fraudshield_auto_refresh",
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                #172554 0%,
                transparent 25%
            ),
            linear-gradient(
                135deg,
                #060b16 0%,
                #0b1120 55%,
                #0f172a 100%
            );

        color: #e5e7eb;
    }

    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 3rem;
        max-width: 1600px;
    }

    [data-testid="stSidebar"] {
        background: #080d18;
        border-right: 1px solid #1e293b;
    }

    [data-testid="stSidebar"] * {
        color: #dbeafe;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .dashboard-header {
        padding: 22px 26px;
        border: 1px solid #1e293b;
        border-radius: 16px;
        background: rgba(15,23,42,0.88);
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,.20);
    }

    .dashboard-title {
        font-size: 32px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -.5px;
    }

    .dashboard-subtitle {
        color: #94a3b8;
        margin-top: 6px;
        font-size: 14px;
    }

    .live-badge {
        display: inline-block;
        background: rgba(34,197,94,.12);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,.35);
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 700;
    }

    .kpi-card {
        background: rgba(15,23,42,.92);
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 18px;
        min-height: 118px;
        box-shadow: 0 6px 20px rgba(0,0,0,.18);
    }

    .kpi-label {
        color: #94a3b8;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .8px;
        font-weight: 700;
    }

    .kpi-value {
        color: #f8fafc;
        font-size: 29px;
        font-weight: 800;
        margin-top: 8px;
    }

    .kpi-detail {
        color: #64748b;
        font-size: 12px;
        margin-top: 5px;
    }

    .danger-value {
        color: #fb7185;
    }

    .success-value {
        color: #4ade80;
    }

    .warning-value {
        color: #fbbf24;
    }

    .blue-value {
        color: #60a5fa;
    }

    .purple-value {
        color: #c084fc;
    }

    .section-title {
        color: #f8fafc;
        font-size: 18px;
        font-weight: 750;
        margin-top: 18px;
        margin-bottom: 12px;
    }

    .alert-panel {
        border: 1px solid rgba(244,63,94,.35);
        background: rgba(127,29,29,.18);
        border-radius: 14px;
        padding: 16px 20px;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    .alert-title {
        color: #fb7185;
        font-size: 16px;
        font-weight: 800;
    }

    .alert-text {
        color: #cbd5e1;
        font-size: 13px;
        margin-top: 5px;
    }

    .healthy-panel {
        border: 1px solid rgba(34,197,94,.30);
        background: rgba(20,83,45,.16);
        border-radius: 14px;
        padding: 16px 20px;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    .case-card {
        background: rgba(15,23,42,.90);
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .case-label {
        color: #64748b;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .7px;
    }

    .case-value {
        color: #f8fafc;
        font-size: 16px;
        font-weight: 700;
        margin-top: 3px;
    }

    .status-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        font-size: 13px;
        font-weight: 700;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #1e293b;
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: #1e293b !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 9px;
        border: 1px solid #334155;
        background: #111827;
        color: #e2e8f0;
    }

    .stButton > button:hover {
        border-color: #3b82f6;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = os.getenv(
    "FRAUD_DB_HOST",
    "localhost",
)

DB_PORT = os.getenv(
    "FRAUD_DB_PORT",
    "5432",
)

DB_NAME = os.getenv(
    "FRAUD_DB_NAME",
    "fraud_detection",
)

DB_USER = os.getenv(
    "FRAUD_DB_USER",
    "saleor",
)

DB_PASSWORD = os.getenv(
    "FRAUD_DB_PASSWORD",
    "saleor",
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        row_factory=dict_row,
    )


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    query = """
        SELECT
            transaction_id,
            customer_id,
            account_id,
            amount,
            channel,
            country,
            fraud_probability,
            predicted_fraud,
            risk_level,
            actual_fraud,
            fraud_type,
            prediction_timestamp

        FROM fraud_predictions

        ORDER BY
            prediction_timestamp DESC;
    """

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(query)

            rows = cursor.fetchall()

    df = pd.DataFrame(rows)

    if not df.empty:

        df["prediction_timestamp"] = (
            pd.to_datetime(
                df["prediction_timestamp"],
                errors="coerce",
                utc=True,
            )
        )

        numeric_columns = [
            "amount",
            "fraud_probability",
            "predicted_fraud",
            "actual_fraud",
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_predictions()

    alert_stats = (
        get_alert_statistics()
    )

    alert_records = (
        get_alerts(
            limit=500
        )
    )

    alerts_df = pd.DataFrame(
        alert_records
    )

except Exception as error:

    st.error(
        f"Unable to connect to FraudShield database: {error}"
    )

    st.stop()


if df.empty:

    st.warning(
        "No prediction data is available yet."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 🛡️ FraudShield"
)

st.sidebar.caption(
    "Fraud Operations Center"
)

st.sidebar.divider()


if st.sidebar.button(
    "↻ Refresh Now"
):

    st.rerun()


st.sidebar.markdown(
    "### Transaction Filters"
)


risk_levels = sorted(
    df["risk_level"]
    .dropna()
    .unique()
    .tolist()
)


channels = sorted(
    df["channel"]
    .dropna()
    .unique()
    .tolist()
)


countries = sorted(
    df["country"]
    .dropna()
    .unique()
    .tolist()
)


selected_risk = (
    st.sidebar.multiselect(
        "Risk Level",
        risk_levels,
        default=risk_levels,
    )
)


selected_channel = (
    st.sidebar.multiselect(
        "Channel",
        channels,
        default=channels,
    )
)


selected_country = (
    st.sidebar.multiselect(
        "Country",
        countries,
        default=countries,
    )
)


filtered_df = df[
    df["risk_level"].isin(
        selected_risk
    )
    &
    df["channel"].isin(
        selected_channel
    )
    &
    df["country"].isin(
        selected_country
    )
].copy()


st.sidebar.divider()

st.sidebar.markdown(
    "### Investigation"
)


status_filter = (
    st.sidebar.multiselect(
        "Case Status",
        [
            "NEW",
            "INVESTIGATING",
            "CONFIRMED_FRAUD",
            "FALSE_POSITIVE",
            "CLOSED",
        ],
        default=[
            "NEW",
            "INVESTIGATING",
        ],
    )
)


st.sidebar.divider()

st.sidebar.caption(
    "ML decision threshold: 40%"
)

st.sidebar.caption(
    "Dashboard refresh: 5 seconds"
)


# ============================================================
# HEADER
# ============================================================

current_time = (
    datetime.now()
    .strftime(
        "%b %d, %Y • %I:%M:%S %p"
    )
)


st.markdown(
    f"""
    <div class="dashboard-header">

        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
        ">

            <div>

                <div class="dashboard-title">
                    FraudShield
                </div>

                <div class="dashboard-subtitle">
                    Real-Time Fraud Intelligence,
                    Detection & Investigation Platform
                </div>

            </div>

            <div style="text-align:right;">

                <span class="live-badge">
                    ● LIVE OPERATIONS
                </span>

                <div style="
                    color:#64748b;
                    font-size:12px;
                    margin-top:8px;
                ">
                    {current_time}
                </div>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TRANSACTION METRICS
# ============================================================

total_transactions = len(
    filtered_df
)


fraud_predictions = int(
    filtered_df[
        "predicted_fraud"
    ].sum()
)


alert_rate = (

    fraud_predictions
    / total_transactions
    * 100

    if total_transactions

    else 0
)


alerted_amount = float(

    filtered_df.loc[

        filtered_df[
            "predicted_fraud"
        ] == 1,

        "amount",

    ].sum()
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

tp = len(

    filtered_df[
        (
            filtered_df[
                "predicted_fraud"
            ] == 1
        )
        &
        (
            filtered_df[
                "actual_fraud"
            ] == 1
        )
    ]
)


fp = len(

    filtered_df[
        (
            filtered_df[
                "predicted_fraud"
            ] == 1
        )
        &
        (
            filtered_df[
                "actual_fraud"
            ] == 0
        )
    ]
)


fn = len(

    filtered_df[
        (
            filtered_df[
                "predicted_fraud"
            ] == 0
        )
        &
        (
            filtered_df[
                "actual_fraud"
            ] == 1
        )
    ]
)


tn = len(

    filtered_df[
        (
            filtered_df[
                "predicted_fraud"
            ] == 0
        )
        &
        (
            filtered_df[
                "actual_fraud"
            ] == 0
        )
    ]
)


precision = (

    tp / (tp + fp)

    if (tp + fp)

    else 0
)


recall = (

    tp / (tp + fn)

    if (tp + fn)

    else 0
)


f1 = (

    2
    * precision
    * recall
    / (precision + recall)

    if (precision + recall)

    else 0
)


accuracy = (

    (tp + tn)
    / (
        tp
        + tn
        + fp
        + fn
    )

    if (
        tp
        + tn
        + fp
        + fn
    )

    else 0
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Executive Overview'
    '</div>',
    unsafe_allow_html=True,
)


k1, k2, k3, k4, k5, k6 = (
    st.columns(6)
)


def kpi_card(
    container,
    label,
    value,
    detail,
    value_class="blue-value",
):

    with container:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    {label}
                </div>

                <div class="
                    kpi-value
                    {value_class}
                ">
                    {value}
                </div>

                <div class="kpi-detail">
                    {detail}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


kpi_card(
    k1,
    "Transactions",
    f"{total_transactions:,}",
    "Unique transactions analyzed",
)


kpi_card(
    k2,
    "Fraud Predictions",
    f"{fraud_predictions:,}",
    "Transactions above 40%",
    "danger-value",
)


kpi_card(
    k3,
    "Alert Rate",
    f"{alert_rate:.2f}%",
    "Model alert frequency",
    "warning-value",
)


kpi_card(
    k4,
    "Amount at Risk",
    f"${alerted_amount:,.0f}",
    "Value of flagged transactions",
    "danger-value",
)


kpi_card(
    k5,
    "Open Cases",
    (
        f"{int(alert_stats['new_alerts']) + int(alert_stats['investigating_alerts']):,}"
    ),
    "New + investigating",
    "purple-value",
)


kpi_card(
    k6,
    "Model Accuracy",
    f"{accuracy:.1%}",
    "Current labeled dataset",
    "success-value",
)


# ============================================================
# ACTIVE ALERT PANEL
# ============================================================

open_cases = (

    int(
        alert_stats[
            "new_alerts"
        ]
    )

    +

    int(
        alert_stats[
            "investigating_alerts"
        ]
    )
)


if open_cases:

    st.markdown(
        f"""
        <div class="alert-panel">

            <div class="alert-title">
                ⚠ ACTIVE INVESTIGATION QUEUE
            </div>

            <div class="alert-text">

                {open_cases} fraud case(s)
                currently require analyst attention.

                {alert_stats['critical_alerts']}
                critical-risk case(s) are present
                in the alert repository.

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class="healthy-panel">

            <b style="color:#4ade80;">
                ● INVESTIGATION QUEUE CLEAR
            </b>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHART THEME
# ============================================================

PLOT_BG = "rgba(0,0,0,0)"
GRID = "#1e293b"
TEXT = "#94a3b8"


def style_chart(fig):

    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(
            color=TEXT
        ),
        margin=dict(
            l=20,
            r=20,
            t=45,
            b=20,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        ),
    )

    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
    )

    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
    )

    return fig


# ============================================================
# RISK + MODEL HEALTH
# ============================================================

left, right = st.columns(
    [1.5, 1]
)


with left:

    st.markdown(
        '<div class="section-title">'
        'Transaction Risk Distribution'
        '</div>',
        unsafe_allow_html=True,
    )

    risk_counts = (

        filtered_df[
            "risk_level"
        ]

        .value_counts()

        .reindex(
            [
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW",
            ],
            fill_value=0,
        )

        .reset_index()
    )

    risk_counts.columns = [
        "Risk Level",
        "Transactions",
    ]

    fig_risk = px.bar(
        risk_counts,
        x="Risk Level",
        y="Transactions",
        color="Risk Level",
        color_discrete_map={
            "CRITICAL": "#ef4444",
            "HIGH": "#f97316",
            "MEDIUM": "#eab308",
            "LOW": "#22c55e",
        },
    )

    fig_risk.update_layout(
        showlegend=False
    )

    style_chart(
        fig_risk
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True,
    )


with right:

    st.markdown(
        '<div class="section-title">'
        'Fraud Detection Recall'
        '</div>',
        unsafe_allow_html=True,
    )

    fig_gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=recall * 100,

            number={
                "suffix": "%",
                "font": {
                    "color": "#f8fafc"
                },
            },

            title={
                "text": "Fraud Recall",
                "font": {
                    "color": "#94a3b8"
                },
            },

            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#64748b",
                },

                "bar": {
                    "color": "#3b82f6"
                },

                "bgcolor": "#111827",

                "bordercolor": "#1e293b",

                "steps": [
                    {
                        "range": [0, 50],
                        "color": "#3f1721",
                    },
                    {
                        "range": [50, 75],
                        "color": "#422006",
                    },
                    {
                        "range": [75, 100],
                        "color": "#052e16",
                    },
                ],
            },
        )
    )

    fig_gauge.update_layout(
        paper_bgcolor=PLOT_BG,
        font={
            "color": TEXT
        },
        height=300,
        margin=dict(
            l=30,
            r=30,
            t=45,
            b=20,
        ),
    )

    st.plotly_chart(
        fig_gauge,
        use_container_width=True,
    )


# ============================================================
# MODEL METRICS
# ============================================================

m1, m2, m3, m4, m5 = (
    st.columns(5)
)


m1.metric(
    "Precision",
    f"{precision:.1%}",
)

m2.metric(
    "Recall",
    f"{recall:.1%}",
)

m3.metric(
    "F1 Score",
    f"{f1:.1%}",
)

m4.metric(
    "False Positives",
    fp,
)

m5.metric(
    "Missed Fraud",
    fn,
)


# ============================================================
# FRAUD PROBABILITY TREND
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Real-Time Fraud Probability'
    '</div>',
    unsafe_allow_html=True,
)


trend = (

    filtered_df[
        [
            "prediction_timestamp",
            "fraud_probability",
        ]
    ]

    .dropna()

    .sort_values(
        "prediction_timestamp"
    )
)


trend["Probability %"] = (

    trend[
        "fraud_probability"
    ]

    * 100
)


fig_trend = px.line(
    trend,
    x="prediction_timestamp",
    y="Probability %",
)


fig_trend.update_traces(
    line=dict(
        color="#60a5fa",
        width=2,
    )
)


fig_trend.add_hline(
    y=40,
    line_dash="dash",
    line_color="#ef4444",
    annotation_text="40% fraud threshold",
)


style_chart(
    fig_trend
)


fig_trend.update_layout(
    xaxis_title="Time",
    yaxis_title="Fraud Probability (%)",
)


st.plotly_chart(
    fig_trend,
    use_container_width=True,
)


# ============================================================
# INVESTIGATION OPERATIONS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔎 Fraud Investigation Center'
    '</div>',
    unsafe_allow_html=True,
)


if alerts_df.empty:

    st.info(
        "No investigation alerts exist."
    )

else:

    queue = alerts_df.copy()

    if status_filter:

        queue = queue[
            queue[
                "alert_status"
            ].isin(
                status_filter
            )
        ]


    if queue.empty:

        st.info(
            "No cases match the selected status filter."
        )

    else:

        queue[
            "Fraud Probability"
        ] = (

            pd.to_numeric(
                queue[
                    "fraud_probability"
                ],
                errors="coerce",
            )

            * 100
        ).round(2)


        queue_display = queue[
            [
                "alert_id",
                "transaction_id",
                "customer_id",
                "Fraud Probability",
                "risk_level",
                "fraud_type",
                "alert_status",
                "analyst_name",
                "created_at",
            ]
        ].copy()


        queue_display.columns = [
            "Case",
            "Transaction",
            "Customer",
            "Fraud %",
            "Risk",
            "Fraud Type",
            "Status",
            "Analyst",
            "Created",
        ]


        st.dataframe(
            queue_display,
            use_container_width=True,
            hide_index=True,
            column_config={

                "Fraud %":
                    st.column_config.ProgressColumn(
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    )
            },
        )


        # ====================================================
        # CASE SELECTION
        # ====================================================

        case_options = {}

        for _, row in queue.iterrows():

            label = (

                f"Case #{row['alert_id']} • "
                f"{row['transaction_id']} • "
                f"{row['risk_level']} • "
                f"{row['alert_status']}"
            )

            case_options[
                label
            ] = row


        selected_case_label = (
            st.selectbox(
                "Select investigation case",
                list(
                    case_options.keys()
                ),
            )
        )


        selected_case = (
            case_options[
                selected_case_label
            ]
        )


        transaction_id = (
            selected_case[
                "transaction_id"
            ]
        )


        # ====================================================
        # CASE DETAILS
        # ====================================================

        st.markdown(
            "#### Case Details"
        )


        c1, c2, c3, c4 = (
            st.columns(4)
        )


        with c1:

            st.markdown(
                f"""
                <div class="case-card">

                    <div class="case-label">
                        Transaction
                    </div>

                    <div class="case-value">
                        {transaction_id}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


        with c2:

            st.markdown(
                f"""
                <div class="case-card">

                    <div class="case-label">
                        Risk Level
                    </div>

                    <div class="case-value">
                        {selected_case['risk_level']}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


        with c3:

            probability = float(
                selected_case[
                    "fraud_probability"
                ]
            )

            st.markdown(
                f"""
                <div class="case-card">

                    <div class="case-label">
                        Fraud Probability
                    </div>

                    <div class="case-value">
                        {probability:.2%}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


        with c4:

            st.markdown(
                f"""
                <div class="case-card">

                    <div class="case-label">
                        Case Status
                    </div>

                    <div class="case-value">
                        {selected_case['alert_status']}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


        # ====================================================
        # TRANSACTION CONTEXT
        # ====================================================

        transaction_context = df[
            df[
                "transaction_id"
            ] == transaction_id
        ]


        if not transaction_context.empty:

            txn = (
                transaction_context
                .iloc[0]
            )

            d1, d2, d3, d4 = (
                st.columns(4)
            )

            d1.metric(
                "Amount",
                f"${float(txn['amount']):,.2f}",
            )

            d2.metric(
                "Channel",
                str(
                    txn["channel"]
                ),
            )

            d3.metric(
                "Country",
                str(
                    txn["country"]
                ),
            )

            d4.metric(
                "Fraud Type",
                str(
                    txn["fraud_type"]
                ),
            )


        # ====================================================
        # ANALYST WORKSPACE
        # ====================================================

        st.markdown(
            "#### Analyst Workspace"
        )


        analyst_name = (
            st.text_input(
                "Analyst Name",
                value=(
                    selected_case[
                        "analyst_name"
                    ]
                    or ""
                ),
                key=(
                    f"analyst_{transaction_id}"
                ),
            )
        )


        existing_notes = (

            selected_case[
                "analyst_notes"
            ]

            or ""
        )


        analyst_notes = (
            st.text_area(
                "Investigation Notes",
                value=existing_notes,
                height=140,
                key=(
                    f"notes_{transaction_id}"
                ),
                placeholder=(
                    "Document evidence, customer verification, "
                    "transaction behavior and investigation findings..."
                ),
            )
        )


        a1, a2, a3 = (
            st.columns(3)
        )


        with a1:

            if st.button(
                "Assign & Investigate",
                key=(
                    f"assign_{transaction_id}"
                ),
            ):

                if not analyst_name.strip():

                    st.warning(
                        "Enter an analyst name first."
                    )

                else:

                    assign_alert(
                        transaction_id,
                        analyst_name.strip(),
                    )

                    if analyst_notes.strip():

                        update_analyst_notes(
                            transaction_id,
                            analyst_notes.strip(),
                        )

                    st.success(
                        "Case assigned and moved to INVESTIGATING."
                    )

                    st.rerun()


        with a2:

            if st.button(
                "Save Notes",
                key=(
                    f"save_{transaction_id}"
                ),
            ):

                update_analyst_notes(
                    transaction_id,
                    analyst_notes.strip(),
                )

                st.success(
                    "Investigation notes saved."
                )

                st.rerun()


        with a3:

            new_status = (
                st.selectbox(
                    "Case Decision",
                    [
                        "INVESTIGATING",
                        "CONFIRMED_FRAUD",
                        "FALSE_POSITIVE",
                        "CLOSED",
                    ],
                    key=(
                        f"status_{transaction_id}"
                    ),
                )
            )


        if st.button(
            "Update Case Decision",
            key=(
                f"decision_{transaction_id}"
            ),
        ):

            update_alert(
                transaction_id=transaction_id,
                status=new_status,
                analyst_name=(
                    analyst_name.strip()
                    or None
                ),
                analyst_notes=(
                    analyst_notes.strip()
                    or None
                ),
            )

            st.success(
                f"Case updated to {new_status}."
            )

            st.rerun()


# ============================================================
# LIVE FRAUD ALERTS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🚨 Live Fraud Signals'
    '</div>',
    unsafe_allow_html=True,
)


live_alerts = (

    filtered_df[
        filtered_df[
            "predicted_fraud"
        ] == 1
    ]

    .sort_values(
        "fraud_probability",
        ascending=False,
    )

    .copy()
)


if live_alerts.empty:

    st.success(
        "No fraud signals for current filters."
    )

else:

    live_alerts[
        "Fraud Probability"
    ] = (

        live_alerts[
            "fraud_probability"
        ]

        * 100
    ).round(2)


    live_table = live_alerts[
        [
            "prediction_timestamp",
            "transaction_id",
            "customer_id",
            "amount",
            "channel",
            "country",
            "Fraud Probability",
            "risk_level",
            "fraud_type",
        ]
    ].copy()


    live_table.columns = [
        "Timestamp",
        "Transaction",
        "Customer",
        "Amount",
        "Channel",
        "Country",
        "Fraud %",
        "Risk",
        "Fraud Type",
    ]


    st.dataframe(
        live_table,
        use_container_width=True,
        hide_index=True,
        column_config={

            "Amount":
                st.column_config.NumberColumn(
                    format="$%.2f"
                ),

            "Fraud %":
                st.column_config.ProgressColumn(
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
        },
    )


# ============================================================
# FRAUD BREAKDOWN
# ============================================================

b1, b2 = st.columns(2)


with b1:

    st.markdown(
        '<div class="section-title">'
        'Fraud Signals by Channel'
        '</div>',
        unsafe_allow_html=True,
    )

    channel_data = (

        filtered_df[
            filtered_df[
                "predicted_fraud"
            ] == 1
        ]

        .groupby(
            "channel"
        )

        .size()

        .reset_index(
            name="Alerts"
        )
    )


    if not channel_data.empty:

        fig_channel = px.bar(
            channel_data,
            x="channel",
            y="Alerts",
        )

        fig_channel.update_traces(
            marker_color="#f43f5e"
        )

        style_chart(
            fig_channel
        )

        st.plotly_chart(
            fig_channel,
            use_container_width=True,
        )

    else:

        st.info(
            "No fraud signals."
        )


with b2:

    st.markdown(
        '<div class="section-title">'
        'Known Fraud Types'
        '</div>',
        unsafe_allow_html=True,
    )

    fraud_types = (

        filtered_df[
            filtered_df[
                "actual_fraud"
            ] == 1
        ]

        .groupby(
            "fraud_type"
        )

        .size()

        .reset_index(
            name="Transactions"
        )
    )


    if not fraud_types.empty:

        fig_type = px.pie(
            fraud_types,
            names="fraud_type",
            values="Transactions",
            hole=.55,
        )

        style_chart(
            fig_type
        )

        st.plotly_chart(
            fig_type,
            use_container_width=True,
        )

    else:

        st.info(
            "No labeled fraud records."
        )


# ============================================================
# CLASSIFICATION SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Model Classification Summary'
    '</div>',
    unsafe_allow_html=True,
)


q1, q2, q3, q4 = (
    st.columns(4)
)


q1.metric(
    "True Positives",
    tp,
)

q2.metric(
    "False Positives",
    fp,
)

q3.metric(
    "False Negatives",
    fn,
)

q4.metric(
    "True Negatives",
    tn,
)


# ============================================================
# COMPLETE TRANSACTION FEED
# ============================================================

with st.expander(
    "View Complete Transaction Feed"
):

    complete = (
        filtered_df.copy()
    )

    complete[
        "Fraud Probability"
    ] = (

        complete[
            "fraud_probability"
        ]

        * 100
    ).round(2)


    transaction_table = complete[
        [
            "prediction_timestamp",
            "transaction_id",
            "customer_id",
            "account_id",
            "amount",
            "channel",
            "country",
            "Fraud Probability",
            "risk_level",
            "predicted_fraud",
            "actual_fraud",
            "fraud_type",
        ]
    ].copy()


    transaction_table.columns = [
        "Timestamp",
        "Transaction",
        "Customer",
        "Account",
        "Amount",
        "Channel",
        "Country",
        "Fraud %",
        "Risk",
        "Predicted Fraud",
        "Actual Fraud",
        "Fraud Type",
    ]


    st.dataframe(
        transaction_table,
        use_container_width=True,
        hide_index=True,
        column_config={

            "Amount":
                st.column_config.NumberColumn(
                    format="$%.2f"
                ),

            "Fraud %":
                st.column_config.ProgressColumn(
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
        },
    )


# ============================================================
# PLATFORM STATUS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Platform Components'
    '</div>',
    unsafe_allow_html=True,
)


s1, s2, s3, s4, s5 = (
    st.columns(5)
)


with s1:

    st.markdown(
        '<div class="status-card">'
        '📡 Kafka<br>Event Streaming'
        '</div>',
        unsafe_allow_html=True,
    )


with s2:

    st.markdown(
        '<div class="status-card">'
        '🧠 Random Forest<br>ML Inference'
        '</div>',
        unsafe_allow_html=True,
    )


with s3:

    st.markdown(
        '<div class="status-card">'
        '🐘 PostgreSQL<br>Prediction Store'
        '</div>',
        unsafe_allow_html=True,
    )


with s4:

    st.markdown(
        '<div class="status-card">'
        '🚨 Alert Manager<br>Case Workflow'
        '</div>',
        unsafe_allow_html=True,
    )


with s5:

    st.markdown(
        '<div class="status-card">'
        '📊 Streamlit<br>Operations UI'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.markdown(
    """
    <div style="
        text-align:center;
        color:#475569;
        font-size:12px;
        padding:10px;
    ">

        FRAUDSHIELD • REAL-TIME FRAUD OPERATIONS PLATFORM
        <br>

        Apache Kafka • Python • Random Forest •
        PostgreSQL • Streamlit

    </div>
    """,
    unsafe_allow_html=True,
)