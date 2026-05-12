from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import run_clustering


st.set_page_config(
    page_title="Parcl Buyer Intelligence",
    page_icon="PI",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
      --ink: #17202a;
      --muted: #617080;
      --line: #e6ebef;
      --panel: #ffffff;
      --accent: #0f766e;
      --soft: #f6f8f9;
    }
    .stApp { background: #f7f9fa; color: var(--ink); }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    h1, h2, h3 { letter-spacing: 0 !important; color: var(--ink); }
    div[data-testid="stMetric"] {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      box-shadow: 0 1px 2px rgba(23,32,42,.04);
    }
    div[data-testid="stMetricLabel"] p { color: var(--muted); }
    .insight {
      background: #ffffff;
      border: 1px solid #e6ebef;
      border-radius: 8px;
      padding: 14px 16px;
      min-height: 118px;
    }
    .insight b { color: #0f766e; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_artifacts(k: int):
    artifacts = run_clustering(k=k)
    return (
        artifacts.data,
        artifacts.elbow_scores,
        artifacts.silhouette_scores,
        artifacts.cluster_profile,
    )


st.title("Parcl Buyer Segmentation & Investment Profiling")
st.caption("Machine learning market intelligence dashboard using K-Means and hierarchical clustering.")

with st.sidebar:
    st.header("Controls")
    selected_k = st.slider("Number of buyer segments", min_value=2, max_value=8, value=4)

data, elbow_scores, silhouette_scores, profile = load_artifacts(selected_k)

with st.sidebar:
    countries = st.multiselect("Country", sorted(data["country"].unique()))
    regions = st.multiselect("Region", sorted(data["region"].unique()))
    purposes = st.multiselect("Acquisition purpose", sorted(data["acquisition_purpose"].unique()))
    client_types = st.multiselect("Client type", sorted(data["client_type"].unique()))
    segments = st.multiselect("Segment", sorted(data["segment"].unique()))

filtered = data.copy()
if countries:
    filtered = filtered[filtered["country"].isin(countries)]
if regions:
    filtered = filtered[filtered["region"].isin(regions)]
if purposes:
    filtered = filtered[filtered["acquisition_purpose"].isin(purposes)]
if client_types:
    filtered = filtered[filtered["client_type"].isin(client_types)]
if segments:
    filtered = filtered[filtered["segment"].isin(segments)]

sales_total = filtered["total_spend"].sum()
investment_rate = filtered["investment_flag"].mean() * 100 if len(filtered) else 0
loan_rate = filtered["loan_flag"].mean() * 100 if len(filtered) else 0

metric_cols = st.columns(4)
metric_cols[0].metric("Clients", f"{len(filtered):,}")
metric_cols[1].metric("Total Sales Linked", f"${sales_total:,.0f}")
metric_cols[2].metric("Investment Buyers", f"{investment_rate:.1f}%")
metric_cols[3].metric("Loan Applied", f"{loan_rate:.1f}%")

tabs = st.tabs(
    [
        "Buyer Segmentation Overview",
        "Investor Behavior Dashboard",
        "Geographic Buyer Analysis",
        "Segment Insights",
        "Model Evaluation",
    ]
)

palette = ["#0f766e", "#334155", "#d97706", "#7c3aed", "#2563eb", "#be123c", "#4d7c0f", "#9333ea"]

with tabs[0]:
    left, right = st.columns([1.05, 1])
    segment_counts = (
        filtered.groupby("segment", as_index=False)
        .agg(clients=("client_id", "count"), avg_spend=("total_spend", "mean"))
        .sort_values("clients", ascending=False)
    )
    left.plotly_chart(
        px.bar(
            segment_counts,
            x="segment",
            y="clients",
            color="segment",
            color_discrete_sequence=palette,
            title="Cluster Distribution",
            labels={"segment": "", "clients": "Clients"},
        ),
        use_container_width=True,
    )
    right.plotly_chart(
        px.scatter(
            filtered,
            x="age",
            y="total_spend",
            color="segment",
            size="property_count",
            hover_data=["client_id", "country", "region", "acquisition_purpose"],
            color_discrete_sequence=palette,
            title="Buyer Positioning by Age and Spend",
            labels={"age": "Age", "total_spend": "Total spend"},
        ),
        use_container_width=True,
    )

with tabs[1]:
    c1, c2 = st.columns(2)
    invest = (
        filtered.groupby(["segment", "acquisition_purpose"], as_index=False)
        .size()
        .rename(columns={"size": "clients"})
    )
    c1.plotly_chart(
        px.bar(
            invest,
            x="segment",
            y="clients",
            color="acquisition_purpose",
            barmode="group",
            title="Investment vs Personal Use by Segment",
            color_discrete_sequence=["#0f766e", "#d97706", "#334155"],
        ),
        use_container_width=True,
    )
    loan = filtered.groupby(["segment", "loan_applied"], as_index=False).size().rename(columns={"size": "clients"})
    c2.plotly_chart(
        px.bar(
            loan,
            x="segment",
            y="clients",
            color="loan_applied",
            barmode="stack",
            title="Financing Behavior by Segment",
            color_discrete_sequence=["#334155", "#0f766e"],
        ),
        use_container_width=True,
    )
    spend = (
        filtered.groupby("segment", as_index=False)
        .agg(avg_sale_price=("avg_sale_price", "mean"), avg_area=("avg_floor_area_sqft", "mean"))
    )
    st.plotly_chart(
        px.scatter(
            spend,
            x="avg_area",
            y="avg_sale_price",
            color="segment",
            size="avg_sale_price",
            title="Average Unit Area vs Sale Price",
            labels={"avg_area": "Average floor area", "avg_sale_price": "Average sale price"},
            color_discrete_sequence=palette,
        ),
        use_container_width=True,
    )

with tabs[2]:
    geo = (
        filtered.groupby(["country", "region", "segment"], as_index=False)
        .agg(clients=("client_id", "count"), sales=("total_spend", "sum"))
    )
    g1, g2 = st.columns([1, 1])
    g1.plotly_chart(
        px.treemap(
            geo,
            path=["country", "region", "segment"],
            values="clients",
            color="segment",
            color_discrete_sequence=palette,
            title="Buyer Segment Map by Country and Region",
        ),
        use_container_width=True,
    )
    top_regions = (
        geo.groupby("region", as_index=False)["clients"]
        .sum()
        .sort_values("clients", ascending=False)
        .head(12)
    )
    g2.plotly_chart(
        px.bar(
            top_regions,
            x="clients",
            y="region",
            orientation="h",
            title="Top Buyer Regions",
            color="clients",
            color_continuous_scale=["#d9eeea", "#0f766e"],
        ),
        use_container_width=True,
    )

with tabs[3]:
    st.subheader("Segment Insights Panel")
    filtered_profile = (
        filtered.groupby(["cluster", "segment"], as_index=False)
        .agg(
            clients=("client_id", "count"),
            avg_age=("age", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            investment_rate=("investment_flag", "mean"),
            loan_rate=("loan_flag", "mean"),
            avg_total_spend=("total_spend", "mean"),
            top_region=("region", lambda x: x.mode().iat[0] if not x.mode().empty else "Unknown"),
        )
        .sort_values("clients", ascending=False)
    )
    insight_cols = st.columns(2)
    for i, row in filtered_profile.iterrows():
        with insight_cols[i % 2]:
            st.markdown(
                f"""
                <div class="insight">
                  <b>{row['segment']}</b><br>
                  {int(row['clients'])} clients | Avg age {row['avg_age']:.1f} | Satisfaction {row['avg_satisfaction']:.2f}<br>
                  Investment share {row['investment_rate'] * 100:.1f}% | Loan share {row['loan_rate'] * 100:.1f}%<br>
                  Avg linked spend ${row['avg_total_spend']:,.0f} | Core region {row['top_region']}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
    st.dataframe(filtered_profile, use_container_width=True, hide_index=True)

with tabs[4]:
    m1, m2 = st.columns(2)
    m1.plotly_chart(
        px.line(elbow_scores, x="k", y="inertia", markers=True, title="Elbow Method"),
        use_container_width=True,
    )
    m2.plotly_chart(
        px.line(
            silhouette_scores,
            x="k",
            y="silhouette_score",
            markers=True,
            title="Silhouette Score",
        ),
        use_container_width=True,
    )
    st.dataframe(profile, use_container_width=True, hide_index=True)
