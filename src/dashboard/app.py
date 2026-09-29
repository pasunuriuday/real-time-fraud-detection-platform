import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
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
    key="fraudshield_refresh",
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
# CSS
# ============================================================

st.markdown(
    """
    <style>
/* =========================================================
   TEXT CONTRAST FIX
   ========================================================= */

.stApp,
.stApp p,
.stApp label {
    color: #cbd5e1;
}

/* Main headings */
.stApp h1 {
    color: #f8fafc !important;
    font-weight: 800 !important;
}

.stApp h2,
.stApp h3 {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}

/* Captions */
[data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
}

/* Sidebar normal text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] p {
    color: #94a3b8 !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

/* Select/multiselect labels */
[data-testid="stWidgetLabel"] p {
    color: #cbd5e1 !important;
}

    </style>
    """,
    unsafe_allow_html=True,
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

        ORDER BY prediction_timestamp DESC;
    """

    with get_connection() as connection:

        return pd.read_sql_query(
            query,
            connection,
        )


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_predictions()

    alerts = get_alerts(
        limit=500
    )

    alerts_df = pd.DataFrame(
        alerts
    )

    alert_stats = (
        get_alert_statistics()
    )

except Exception as error:

    st.error(
        f"Database connection failed: {error}"
    )

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

if df.empty:

    st.warning(
        "No fraud prediction data available."
    )

    st.stop()


df["prediction_timestamp"] = (
    pd.to_datetime(
        df["prediction_timestamp"],
        errors="coerce",
        utc=True,
    )
)


for column in [
    "amount",
    "fraud_probability",
    "predicted_fraud",
    "actual_fraud",
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🛡️ FraudShield"
)

st.sidebar.caption(
    "Fraud Operations Center"
)

st.sidebar.divider()


if st.sidebar.button(
    "↻ Refresh Dashboard",
    use_container_width=True,
):

    st.rerun()


st.sidebar.subheader(
    "Transaction Filters"
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


selected_channels = (
    st.sidebar.multiselect(
        "Channel",
        channels,
        default=channels,
    )
)


selected_countries = (
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
        selected_channels
    )
    &
    df["country"].isin(
        selected_countries
    )
].copy()


st.sidebar.divider()

st.sidebar.subheader(
    "Investigation Filters"
)


selected_statuses = (
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
    "ML threshold: 40%"
)

st.sidebar.caption(
    "Auto refresh: 5 seconds"
)


# ============================================================
# HEADER
# ============================================================

header_left, header_right = (
    st.columns(
        [4, 1]
    )
)


with header_left:

    st.title(
        "🛡️ FraudShield"
    )

    st.caption(
        "Real-Time Fraud Intelligence, "
        "Detection & Investigation Platform"
    )


with header_right:

    st.success(
        "● LIVE OPERATIONS"
    )

    st.caption(
        datetime.now().strftime(
            "%b %d, %Y • %I:%M:%S %p"
        )
    )


st.divider()


# ============================================================
# EXECUTIVE METRICS
# ============================================================

st.subheader(
    "Executive Overview"
)


total_transactions = len(
    filtered_df
)


fraud_predictions = int(
    filtered_df[
        "predicted_fraud"
    ]
    .fillna(0)
    .sum()
)


fraud_rate = (

    fraud_predictions
    / total_transactions
    * 100

    if total_transactions

    else 0
)


amount_at_risk = float(

    filtered_df.loc[
        filtered_df[
            "predicted_fraud"
        ] == 1,
        "amount",
    ]

    .fillna(0)

    .sum()
)


new_alerts = int(
    alert_stats.get(
        "new_alerts",
        0,
    )
)


investigating_alerts = int(
    alert_stats.get(
        "investigating_alerts",
        0,
    )
)


open_cases = (
    new_alerts
    +
    investigating_alerts
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
    /
    (
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
# KPI ROW
# ============================================================

k1, k2, k3, k4, k5, k6 = (
    st.columns(6)
)


k1.metric(
    "Transactions",
    f"{total_transactions:,}",
)


k2.metric(
    "Fraud Predictions",
    f"{fraud_predictions:,}",
)


k3.metric(
    "Alert Rate",
    f"{fraud_rate:.2f}%",
)


k4.metric(
    "Amount at Risk",
    f"${amount_at_risk:,.0f}",
)


k5.metric(
    "Open Cases",
    f"{open_cases:,}",
)


k6.metric(
    "Model Accuracy",
    f"{accuracy:.1%}",
)


# ============================================================
# QUEUE WARNING
# ============================================================

if open_cases > 0:

    st.warning(
        f"{open_cases} fraud investigation case(s) "
        "currently require analyst attention."
    )

else:

    st.success(
        "Investigation queue is clear."
    )


# ============================================================
# RISK DISTRIBUTION + RECALL
# ============================================================

st.divider()

chart_left, chart_right = (
    st.columns(
        [1.6, 1]
    )
)


with chart_left:

    st.subheader(
        "Transaction Risk Distribution"
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


    risk_chart = px.bar(
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


    risk_chart.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )


    st.plotly_chart(
        risk_chart,
        use_container_width=True,
    )


with chart_right:

    st.subheader(
        "Fraud Detection Recall"
    )


    recall_chart = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=recall * 100,

            number={
                "suffix": "%"
            },

            gauge={
                "axis": {
                    "range": [
                        0,
                        100,
                    ]
                },

                "bar": {
                    "color": "#3b82f6"
                },

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


    recall_chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )


    st.plotly_chart(
        recall_chart,
        use_container_width=True,
    )


# ============================================================
# MODEL HEALTH
# ============================================================

st.subheader(
    "Model Health"
)


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

st.divider()

st.subheader(
    "Real-Time Fraud Probability"
)


trend_df = (

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

    .copy()
)


trend_df[
    "Fraud Probability (%)"
] = (

    trend_df[
        "fraud_probability"
    ]

    * 100
)


trend_chart = px.line(
    trend_df,
    x="prediction_timestamp",
    y="Fraud Probability (%)",
)


trend_chart.update_traces(
    line={
        "color": "#60a5fa",
        "width": 2,
    }
)


trend_chart.add_hline(
    y=40,
    line_dash="dash",
    line_color="#ef4444",
    annotation_text="40% fraud threshold",
)


trend_chart.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20,
    ),
)


st.plotly_chart(
    trend_chart,
    use_container_width=True,
)


# ============================================================
# INVESTIGATION CENTER
# ============================================================

st.divider()

st.header(
    "🔎 Fraud Investigation Center"
)


if alerts_df.empty:

    st.info(
        "No fraud investigation cases exist."
    )

else:

    queue_df = (
        alerts_df.copy()
    )


    if selected_statuses:

        queue_df = queue_df[
            queue_df[
                "alert_status"
            ].isin(
                selected_statuses
            )
        ]


    if queue_df.empty:

        st.info(
            "No cases match the selected status filters."
        )

    else:

        queue_display = (
            queue_df.copy()
        )


        queue_display[
            "fraud_probability"
        ] = (

            pd.to_numeric(
                queue_display[
                    "fraud_probability"
                ],
                errors="coerce",
            )

            * 100
        )


        queue_display = queue_display[
            [
                "alert_id",
                "transaction_id",
                "customer_id",
                "fraud_probability",
                "risk_level",
                "fraud_type",
                "alert_status",
                "analyst_name",
                "created_at",
            ]
        ]


        queue_display.columns = [
            "Case ID",
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
                        "Fraud Probability",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
            },
        )


        # ====================================================
        # CASE SELECTOR
        # ====================================================

        case_labels = {}


        for _, row in queue_df.iterrows():

            label = (
                f"Case #{row['alert_id']} | "
                f"{row['transaction_id']} | "
                f"{row['risk_level']} | "
                f"{row['alert_status']}"
            )

            case_labels[
                label
            ] = row


        selected_label = (
            st.selectbox(
                "Select investigation case",
                list(
                    case_labels.keys()
                ),
            )
        )


        selected_case = (
            case_labels[
                selected_label
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

        st.subheader(
            "Case Details"
        )


        case1, case2, case3, case4 = (
            st.columns(4)
        )


        case1.metric(
            "Case ID",
            f"#{selected_case['alert_id']}",
        )


        case2.metric(
            "Risk Level",
            selected_case[
                "risk_level"
            ],
        )


        case3.metric(
            "Fraud Probability",
            (
                f"{float(selected_case['fraud_probability']):.2%}"
            ),
        )


        case4.metric(
            "Status",
            selected_case[
                "alert_status"
            ],
        )


        st.code(
            transaction_id,
            language=None,
        )


        # ====================================================
        # TRANSACTION CONTEXT
        # ====================================================

        transaction_rows = df[
            df[
                "transaction_id"
            ] == transaction_id
        ]


        if not transaction_rows.empty:

            transaction = (
                transaction_rows.iloc[0]
            )


            t1, t2, t3, t4 = (
                st.columns(4)
            )


            t1.metric(
                "Amount",
                f"${float(transaction['amount']):,.2f}",
            )


            t2.metric(
                "Channel",
                str(
                    transaction[
                        "channel"
                    ]
                ),
            )


            t3.metric(
                "Country",
                str(
                    transaction[
                        "country"
                    ]
                ),
            )


            t4.metric(
                "Fraud Type",
                str(
                    transaction[
                        "fraud_type"
                    ]
                ),
            )


        # ====================================================
        # ANALYST WORKSPACE
        # ====================================================

        st.subheader(
            "Analyst Workspace"
        )


        existing_analyst = (
            selected_case.get(
                "analyst_name"
            )

            or ""
        )


        existing_notes = (
            selected_case.get(
                "analyst_notes"
            )

            or ""
        )


        analyst_name = (
            st.text_input(
                "Analyst Name",
                value=existing_analyst,
                key=(
                    f"analyst_{transaction_id}"
                ),
            )
        )


        analyst_notes = (
            st.text_area(
                "Investigation Notes",
                value=existing_notes,
                height=150,
                placeholder=(
                    "Document transaction review, "
                    "customer verification, evidence "
                    "and investigation findings."
                ),
                key=(
                    f"notes_{transaction_id}"
                ),
            )
        )


        action1, action2 = (
            st.columns(2)
        )


        with action1:

            if st.button(
                "👤 Assign & Start Investigation",
                use_container_width=True,
                key=(
                    f"assign_{transaction_id}"
                ),
            ):

                if not analyst_name.strip():

                    st.warning(
                        "Enter an analyst name."
                    )

                else:

                    success = assign_alert(
                        transaction_id,
                        analyst_name.strip(),
                    )


                    if (
                        success
                        and
                        analyst_notes.strip()
                    ):

                        update_analyst_notes(
                            transaction_id,
                            analyst_notes.strip(),
                        )


                    if success:

                        st.success(
                            "Case assigned and moved to INVESTIGATING."
                        )

                        st.rerun()

                    else:

                        st.error(
                            "Case could not be found."
                        )


        with action2:

            if st.button(
                "💾 Save Investigation Notes",
                use_container_width=True,
                key=(
                    f"save_notes_{transaction_id}"
                ),
            ):

                success = (
                    update_analyst_notes(
                        transaction_id,
                        analyst_notes.strip(),
                    )
                )


                if success:

                    st.success(
                        "Investigation notes saved."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Case could not be found."
                    )


        st.markdown(
            "### Case Decision"
        )


        decision = (
            st.selectbox(
                "Decision",
                [
                    "INVESTIGATING",
                    "CONFIRMED_FRAUD",
                    "FALSE_POSITIVE",
                    "CLOSED",
                ],
                key=(
                    f"decision_{transaction_id}"
                ),
            )
        )


        if st.button(
            "Update Case Decision",
            type="primary",
            use_container_width=True,
            key=(
                f"update_{transaction_id}"
            ),
        ):

            success = update_alert(
                transaction_id=transaction_id,
                status=decision,
                analyst_name=(
                    analyst_name.strip()
                    or None
                ),
                analyst_notes=(
                    analyst_notes.strip()
                    or None
                ),
            )


            if success:

                st.success(
                    f"Case status changed to {decision}."
                )

                st.rerun()

            else:

                st.error(
                    "Case could not be found."
                )


# ============================================================
# LIVE FRAUD SIGNALS
# ============================================================

st.divider()

st.header(
    "🚨 Live Fraud Signals"
)


fraud_df = (

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


if fraud_df.empty:

    st.success(
        "No fraud signals match the current filters."
    )

else:

    fraud_display = (
        fraud_df.copy()
    )


    fraud_display[
        "fraud_probability"
    ] = (

        fraud_display[
            "fraud_probability"
        ]

        * 100
    )


    fraud_display = fraud_display[
        [
            "prediction_timestamp",
            "transaction_id",
            "customer_id",
            "amount",
            "channel",
            "country",
            "fraud_probability",
            "risk_level",
            "fraud_type",
        ]
    ]


    fraud_display.columns = [
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
        fraud_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Amount":
                st.column_config.NumberColumn(
                    "Amount",
                    format="$%.2f",
                ),

            "Fraud %":
                st.column_config.ProgressColumn(
                    "Fraud Probability",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
        },
    )


# ============================================================
# FRAUD ANALYTICS
# ============================================================

st.divider()


analytics_left, analytics_right = (
    st.columns(2)
)


with analytics_left:

    st.subheader(
        "Fraud Signals by Channel"
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

        channel_chart = px.bar(
            channel_data,
            x="channel",
            y="Alerts",
        )


        channel_chart.update_traces(
            marker_color="#f43f5e"
        )


        channel_chart.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )


        st.plotly_chart(
            channel_chart,
            use_container_width=True,
        )


    else:

        st.info(
            "No fraud signals."
        )


with analytics_right:

    st.subheader(
        "Known Fraud Types"
    )


    fraud_type_data = (

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


    if not fraud_type_data.empty:

        fraud_type_chart = px.pie(
            fraud_type_data,
            names="fraud_type",
            values="Transactions",
            hole=0.55,
        )


        fraud_type_chart.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
        )


        st.plotly_chart(
            fraud_type_chart,
            use_container_width=True,
        )


    else:

        st.info(
            "No labeled fraud transactions."
        )


# ============================================================
# CLASSIFICATION SUMMARY
# ============================================================

st.subheader(
    "Model Classification Summary"
)


c1, c2, c3, c4 = (
    st.columns(4)
)


c1.metric(
    "True Positives",
    tp,
)


c2.metric(
    "False Positives",
    fp,
)


c3.metric(
    "False Negatives",
    fn,
)


c4.metric(
    "True Negatives",
    tn,
)


# ============================================================
# COMPLETE TRANSACTION FEED
# ============================================================

st.divider()


with st.expander(
    "Complete Transaction Feed"
):

    complete_df = (
        filtered_df.copy()
    )


    complete_df[
        "fraud_probability"
    ] = (

        complete_df[
            "fraud_probability"
        ]

        * 100
    )


    complete_df = complete_df[
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
    ]


    complete_df.columns = [
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
        complete_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Amount":
                st.column_config.NumberColumn(
                    format="$%.2f",
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
# PLATFORM ARCHITECTURE
# ============================================================

st.divider()

st.subheader(
    "Platform Architecture"
)


p1, p2, p3, p4, p5 = (
    st.columns(5)
)


p1.info(
    "📡 KAFKA\n\nEvent Streaming"
)


p2.info(
    "🧠 ML MODEL\n\nRandom Forest"
)


p3.info(
    "🐘 POSTGRESQL\n\nPrediction Store"
)


p4.info(
    "🚨 ALERT MANAGER\n\nCase Workflow"
)


p5.info(
    "📊 STREAMLIT\n\nOperations UI"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FraudShield • Real-Time Fraud Operations Platform | "
    "Apache Kafka • Python • Random Forest • PostgreSQL • Streamlit"
)