import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudShield | Real-Time Fraud Intelligence",
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
# CUSTOM PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background:
            radial-gradient(circle at top right, #172554 0%, transparent 28%),
            linear-gradient(135deg, #060b16 0%, #0b1120 55%, #0f172a 100%);
        color: #e5e7eb;
    }

    /* Main container */
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #080d18;
        border-right: 1px solid #1e293b;
    }

    [data-testid="stSidebar"] * {
        color: #dbeafe;
    }

    /* Remove Streamlit header */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Header container */
    .dashboard-header {
        padding: 20px 24px;
        border: 1px solid #1e293b;
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.82);
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    }

    .dashboard-title {
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        color: #f8fafc;
        letter-spacing: -0.5px;
    }

    .dashboard-subtitle {
        color: #94a3b8;
        margin-top: 6px;
        font-size: 14px;
    }

    .live-badge {
        display: inline-block;
        background: rgba(34,197,94,0.12);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.35);
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* KPI cards */
    .kpi-card {
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 18px;
        min-height: 120px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.18);
    }

    .kpi-label {
        color: #94a3b8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
    }

    .kpi-value {
        color: #f8fafc;
        font-size: 30px;
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

    /* Section header */
    .section-title {
        color: #f8fafc;
        font-size: 18px;
        font-weight: 750;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    /* Alert panel */
    .alert-panel {
        border: 1px solid rgba(244,63,94,0.35);
        background: rgba(127,29,29,0.18);
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

    /* Healthy panel */
    .healthy-panel {
        border: 1px solid rgba(34,197,94,0.30);
        background: rgba(20,83,45,0.16);
        border-radius: 14px;
        padding: 16px 20px;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    /* Status cards */
    .status-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        font-size: 13px;
        font-weight: 700;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #1e293b;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border-color: #1e293b !important;
    }

    /* Button */
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

DB_HOST = os.getenv("FRAUD_DB_HOST", "localhost")
DB_PORT = os.getenv("FRAUD_DB_PORT", "5432")
DB_NAME = os.getenv("FRAUD_DB_NAME", "fraud_detection")
DB_USER = os.getenv("FRAUD_DB_USER", "saleor")
DB_PASSWORD = os.getenv("FRAUD_DB_PASSWORD", "saleor")


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
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_predictions():

    query = """
        SELECT
            id,
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
        ORDER BY prediction_timestamp DESC;
    """

    with get_connection() as connection:
        return pd.read_sql_query(
            query,
            connection,
        )


# ============================================================
# LOAD POSTGRESQL
# ============================================================

try:
    df = load_predictions()

except Exception as error:

    st.error(
        "Unable to connect to the FraudShield PostgreSQL database."
    )

    st.exception(error)
    st.stop()


if df.empty:

    st.warning(
        "No transaction predictions are currently available."
    )

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

df["amount"] = pd.to_numeric(
    df["amount"],
    errors="coerce",
).fillna(0)

df["fraud_probability"] = pd.to_numeric(
    df["fraud_probability"],
    errors="coerce",
).fillna(0)

df["predicted_fraud"] = pd.to_numeric(
    df["predicted_fraud"],
    errors="coerce",
).fillna(0).astype(int)

df["actual_fraud"] = pd.to_numeric(
    df["actual_fraud"],
    errors="coerce",
).fillna(0).astype(int)

df["prediction_timestamp"] = pd.to_datetime(
    df["prediction_timestamp"],
    errors="coerce",
    utc=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 🛡️ FraudShield"
)

st.sidebar.caption(
    "Real-Time Transaction Intelligence"
)

st.sidebar.success(
    "● LIVE MONITORING"
)

st.sidebar.caption(
    "Data refreshes automatically every 5 seconds."
)

if st.sidebar.button(
    "↻ Refresh Now"
):
    st.rerun()


st.sidebar.divider()

st.sidebar.markdown(
    "### Filters"
)


# ============================================================
# FILTERS
# ============================================================

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


selected_risk = st.sidebar.multiselect(
    "Risk Level",
    risk_levels,
    default=risk_levels,
)

selected_channel = st.sidebar.multiselect(
    "Channel",
    channels,
    default=channels,
)

selected_country = st.sidebar.multiselect(
    "Country",
    countries,
    default=countries,
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


# ============================================================
# HEADER
# ============================================================

current_time = datetime.now().strftime(
    "%b %d, %Y • %I:%M:%S %p"
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
                    Real-Time Banking Fraud Intelligence &
                    Investigation Platform
                </div>
            </div>

            <div style="text-align:right;">
                <span class="live-badge">
                    ● LIVE
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
# KPI CALCULATIONS
# ============================================================

total_transactions = len(
    filtered_df
)

fraud_alerts = int(
    filtered_df["predicted_fraud"].sum()
)

actual_fraud = int(
    filtered_df["actual_fraud"].sum()
)

transaction_value = float(
    filtered_df["amount"].sum()
)

alerted_amount = float(
    filtered_df.loc[
        filtered_df["predicted_fraud"] == 1,
        "amount",
    ].sum()
)

alert_rate = (
    fraud_alerts
    / total_transactions
    * 100
    if total_transactions
    else 0
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

tp = len(
    filtered_df[
        (filtered_df["predicted_fraud"] == 1)
        &
        (filtered_df["actual_fraud"] == 1)
    ]
)

fp = len(
    filtered_df[
        (filtered_df["predicted_fraud"] == 1)
        &
        (filtered_df["actual_fraud"] == 0)
    ]
)

fn = len(
    filtered_df[
        (filtered_df["predicted_fraud"] == 0)
        &
        (filtered_df["actual_fraud"] == 1)
    ]
)

tn = len(
    filtered_df[
        (filtered_df["predicted_fraud"] == 0)
        &
        (filtered_df["actual_fraud"] == 0)
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
    2 * precision * recall
    / (precision + recall)
    if (precision + recall)
    else 0
)

accuracy = (
    (tp + tn)
    / (tp + tn + fp + fn)
    if (tp + tn + fp + fn)
    else 0
)


# ============================================================
# KPI CARDS
# ============================================================

st.markdown(
    '<div class="section-title">Executive Overview</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)


with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
                Transactions
            </div>
            <div class="kpi-value blue-value">
                {total_transactions:,}
            </div>
            <div class="kpi-detail">
                Transactions analyzed
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
                Fraud Alerts
            </div>
            <div class="kpi-value danger-value">
                {fraud_alerts:,}
            </div>
            <div class="kpi-detail">
                Model-generated alerts
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
                Alert Rate
            </div>
            <div class="kpi-value warning-value">
                {alert_rate:.2f}%
            </div>
            <div class="kpi-detail">
                Flagged transaction rate
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
                Amount at Risk
            </div>
            <div class="kpi-value danger-value">
                ${alerted_amount:,.0f}
            </div>
            <div class="kpi-detail">
                Value of alerted transactions
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with k5:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
                Model Accuracy
            </div>
            <div class="kpi-value success-value">
                {accuracy:.1%}
            </div>
            <div class="kpi-detail">
                Current labeled sample
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ALERT STATUS
# ============================================================

if fraud_alerts > 0:

    st.markdown(
        f"""
        <div class="alert-panel">
            <div class="alert-title">
                ⚠ ACTIVE FRAUD SIGNALS
            </div>

            <div class="alert-text">
                {fraud_alerts} transaction(s) have crossed
                the fraud detection threshold.
                ${alerted_amount:,.2f} in transaction value
                is currently associated with active alerts.
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
                ● NO ACTIVE FRAUD SIGNALS
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
# RISK DISTRIBUTION + MODEL HEALTH
# ============================================================

left, right = st.columns(
    [1.5, 1]
)


with left:

    st.markdown(
        '<div class="section-title">Transaction Risk Distribution</div>',
        unsafe_allow_html=True,
    )

    risk_counts = (
        filtered_df["risk_level"]
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

    style_chart(fig_risk)

    st.plotly_chart(
        fig_risk,
        use_container_width=True,
    )


with right:

    st.markdown(
        '<div class="section-title">Model Health</div>',
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
# MODEL METRIC STRIP
# ============================================================

m1, m2, m3, m4 = st.columns(4)

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
    "Missed Fraud",
    fn,
)


# ============================================================
# FRAUD PROBABILITY TREND
# ============================================================

st.markdown(
    '<div class="section-title">Real-Time Fraud Probability</div>',
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
    trend["fraud_probability"]
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

# 50% fraud threshold
fig_trend.add_hline(
    y=40,
    line_dash="dash",
    line_color="#ef4444",
    annotation_text="Fraud threshold",
)

style_chart(fig_trend)

fig_trend.update_layout(
    xaxis_title="Time",
    yaxis_title="Fraud Probability (%)",
)

st.plotly_chart(
    fig_trend,
    use_container_width=True,
)


# ============================================================
# CHANNEL + FRAUD TYPE
# ============================================================

c1, c2 = st.columns(2)


with c1:

    st.markdown(
        '<div class="section-title">Fraud Alerts by Channel</div>',
        unsafe_allow_html=True,
    )

    channel_data = (
        filtered_df[
            filtered_df["predicted_fraud"] == 1
        ]
        .groupby("channel")
        .size()
        .reset_index(
            name="Alerts"
        )
    )

    if channel_data.empty:

        st.info(
            "No fraud alerts for current filters."
        )

    else:

        fig_channel = px.bar(
            channel_data,
            x="channel",
            y="Alerts",
        )

        fig_channel.update_traces(
            marker_color="#f43f5e"
        )

        style_chart(fig_channel)

        st.plotly_chart(
            fig_channel,
            use_container_width=True,
        )


with c2:

    st.markdown(
        '<div class="section-title">Known Fraud Types</div>',
        unsafe_allow_html=True,
    )

    fraud_types = (
        filtered_df[
            filtered_df["actual_fraud"] == 1
        ]
        .groupby("fraud_type")
        .size()
        .reset_index(
            name="Transactions"
        )
    )

    if fraud_types.empty:

        st.info(
            "No labeled fraud records."
        )

    else:

        fig_type = px.pie(
            fraud_types,
            names="fraud_type",
            values="Transactions",
            hole=0.55,
        )

        style_chart(fig_type)

        st.plotly_chart(
            fig_type,
            use_container_width=True,
        )


# ============================================================
# LIVE FRAUD ALERTS
# ============================================================

st.markdown(
    '<div class="section-title">🚨 Live Fraud Alerts</div>',
    unsafe_allow_html=True,
)

alerts = (
    filtered_df[
        filtered_df["predicted_fraud"] == 1
    ]
    .sort_values(
        "fraud_probability",
        ascending=False,
    )
    .copy()
)


if alerts.empty:

    st.success(
        "No active fraud alerts."
    )

else:

    alerts["Fraud Probability"] = (
        alerts["fraud_probability"]
        * 100
    ).round(2)

    alert_table = alerts[
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

    alert_table.columns = [
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
        alert_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Amount": st.column_config.NumberColumn(
                format="$%.2f"
            ),
            "Fraud %": st.column_config.ProgressColumn(
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )


# ============================================================
# INVESTIGATION QUEUE
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Investigation Queue</div>',
    unsafe_allow_html=True,
)

investigation = (
    filtered_df[
        (
            filtered_df["risk_level"]
            .isin(
                [
                    "CRITICAL",
                    "HIGH",
                    "MEDIUM",
                ]
            )
        )
        |
        (
            (
                filtered_df["actual_fraud"] == 1
            )
            &
            (
                filtered_df["predicted_fraud"] == 0
            )
        )
    ]
    .sort_values(
        "fraud_probability",
        ascending=False,
    )
    .copy()
)


investigation["Fraud Probability"] = (
    investigation["fraud_probability"]
    * 100
).round(2)


investigation_table = investigation[
    [
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


investigation_table.columns = [
    "Transaction",
    "Customer",
    "Account",
    "Amount",
    "Channel",
    "Country",
    "Fraud %",
    "Risk",
    "Predicted",
    "Actual",
    "Fraud Type",
]


st.dataframe(
    investigation_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Amount": st.column_config.NumberColumn(
            format="$%.2f"
        ),
        "Fraud %": st.column_config.ProgressColumn(
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
    },
)


# ============================================================
# MODEL CONFUSION MATRIX
# ============================================================

st.markdown(
    '<div class="section-title">Model Classification Summary</div>',
    unsafe_allow_html=True,
)

q1, q2, q3, q4 = st.columns(4)

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
# ALL TRANSACTIONS
# ============================================================

with st.expander(
    "View Complete Transaction Feed"
):

    complete = filtered_df.copy()

    complete["fraud_probability"] = (
        complete["fraud_probability"]
        * 100
    ).round(2)

    st.dataframe(
        complete[
            [
                "prediction_timestamp",
                "transaction_id",
                "customer_id",
                "account_id",
                "amount",
                "channel",
                "country",
                "fraud_probability",
                "risk_level",
                "predicted_fraud",
                "actual_fraud",
                "fraud_type",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PLATFORM STATUS
# ============================================================

st.markdown(
    '<div class="section-title">Platform Status</div>',
    unsafe_allow_html=True,
)

s1, s2, s3, s4, s5 = st.columns(5)


with s1:
    st.markdown(
        '<div class="status-card">🟢 Kafka<br>Streaming</div>',
        unsafe_allow_html=True,
    )

with s2:
    st.markdown(
        '<div class="status-card">🟢 Spark<br>Processing</div>',
        unsafe_allow_html=True,
    )

with s3:
    st.markdown(
        '<div class="status-card">🟢 Random Forest<br>Inference</div>',
        unsafe_allow_html=True,
    )

with s4:
    st.markdown(
        '<div class="status-card">🟢 PostgreSQL<br>Storage</div>',
        unsafe_allow_html=True,
    )

with s5:
    st.markdown(
        '<div class="status-card">🟢 Dashboard<br>Live</div>',
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
        FRAUDSHIELD • REAL-TIME FRAUD INTELLIGENCE PLATFORM<br>
        Apache Kafka • Apache Spark • Random Forest •
        PostgreSQL • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)