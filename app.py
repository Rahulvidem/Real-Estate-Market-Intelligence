from __future__ import annotations

import plotly.express as px
import streamlit as st

from pipeline import run_clustering



st.set_page_config(
    page_title="Parcl Buyer Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLORS = ["#00d1b2", "#ffb703", "#8ec5ff", "#ff5c8a", "#7cdd77", "#b185ff", "#f97316", "#38bdf8"]


def style_chart(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.04)",
        font=dict(color="#e6edf7", family="Segoe UI, Arial, sans-serif"),
        title=dict(font=dict(size=19, color="#ffffff"), x=0.02),
        margin=dict(l=25, r=25, t=60, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d8e2ef"),
            orientation="h",
            y=1.08,
            x=1,
            xanchor="right",
        ),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)", color="#d8e2ef")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.12)", color="#d8e2ef")
    return fig


st.markdown(
    """
    <style>
    .stApp {
      background:
        radial-gradient(circle at top left, rgba(0, 209, 178, 0.26), transparent 340px),
        radial-gradient(circle at top right, rgba(142, 197, 255, 0.20), transparent 380px),
        linear-gradient(135deg, #060b16 0%, #0b1220 48%, #111827 100%);
      color: #f8fafc;
    }

    .block-container {
      max-width: 1450px;
      padding-top: 1.2rem;
      padding-bottom: 2.5rem;
    }

    [data-testid="stSidebar"] {
      background: #07101f;
      border-right: 1px solid rgba(255,255,255,0.12);
    }

    [data-testid="stSidebar"] * {
      color: #e6edf7;
    }

    h1, h2, h3 {
      color: #ffffff;
      letter-spacing: 0 !important;
    }

    .hero {
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 24px;
      padding: 30px;
      background:
        linear-gradient(135deg, rgba(15,23,42,0.92), rgba(15,23,42,0.62)),
        radial-gradient(circle at 82% 24%, rgba(0,209,178,0.24), transparent 280px);
      box-shadow: 0 24px 70px rgba(0,0,0,0.35);
      overflow: hidden;
      position: relative;
    }

    .hero:after {
      content: "";
      position: absolute;
      width: 420px;
      height: 420px;
      right: -130px;
      top: -120px;
      border-radius: 90px;
      transform: rotate(28deg);
      background:
        linear-gradient(135deg, rgba(0,209,178,0.20), rgba(142,197,255,0.05));
      border: 1px solid rgba(255,255,255,0.12);
    }

    .hero-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr);
      gap: 24px;
      align-items: center;
      position: relative;
      z-index: 2;
    }

    .badge {
      display: inline-block;
      padding: 7px 12px;
      border-radius: 999px;
      background: rgba(0,209,178,0.12);
      border: 1px solid rgba(0,209,178,0.35);
      color: #8fffea;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .hero-title {
      margin: 18px 0 12px;
      max-width: 900px;
      font-size: clamp(34px, 5vw, 64px);
      line-height: 1;
      font-weight: 800;
      color: #ffffff;
    }

    .hero-title span {
      color: #00d1b2;
    }

    .hero-text {
      max-width: 800px;
      color: #b8c6d8;
      font-size: 16px;
      line-height: 1.65;
    }

    .mini-city {
      height: 260px;
      border-radius: 22px;
      background:
        linear-gradient(90deg, rgba(255,255,255,0.10) 1px, transparent 1px),
        linear-gradient(rgba(255,255,255,0.10) 1px, transparent 1px),
        linear-gradient(145deg, rgba(15,23,42,0.86), rgba(2,6,23,0.65));
      background-size: 34px 34px, 34px 34px, auto;
      border: 1px solid rgba(255,255,255,0.14);
      transform: perspective(900px) rotateX(8deg) rotateY(-10deg);
      box-shadow: 0 28px 70px rgba(0,0,0,0.38);
      position: relative;
      overflow: hidden;
    }

    .bar {
      position: absolute;
      bottom: 34px;
      width: 42px;
      border-radius: 10px 10px 0 0;
      background: linear-gradient(180deg, #8fffea, #00d1b2);
      box-shadow: 14px 14px 30px rgba(0,0,0,0.35);
    }

    .bar.one { left: 54px; height: 105px; }
    .bar.two { left: 116px; height: 165px; background: linear-gradient(180deg, #ffd166, #f97316); }
    .bar.three { left: 178px; height: 132px; background: linear-gradient(180deg, #bfdbfe, #6366f1); }
    .bar.four { left: 240px; height: 82px; background: linear-gradient(180deg, #fda4af, #ff5c8a); }

    .metric-box {
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 18px;
      padding: 18px;
      background: rgba(15,23,42,0.72);
      box-shadow: 0 16px 42px rgba(0,0,0,0.24);
    }

    .metric-label {
      color: #9fb0c4;
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .metric-value {
      margin-top: 10px;
      color: #ffffff;
      font-size: 30px;
      font-weight: 800;
    }

    .metric-note {
      margin-top: 6px;
      color: #9fb0c4;
      font-size: 13px;
    }

    .section-title {
      margin: 22px 0 10px;
      font-size: 22px;
      font-weight: 800;
      color: #ffffff;
    }

    .insight {
      min-height: 142px;
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 18px;
      padding: 16px;
      background: rgba(15,23,42,0.72);
      box-shadow: 0 16px 42px rgba(0,0,0,0.22);
    }

    .insight b {
      color: #8fffea;
      font-size: 17px;
    }

    .insight p {
      color: #b8c6d8;
      font-size: 13px;
      line-height: 1.65;
      margin: 8px 0 0;
    }

    .stTabs [data-baseweb="tab-list"] {
      gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
      background: rgba(15,23,42,0.72);
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.10);
      color: #d8e2ef;
      padding: 8px 14px;
      font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
      background: rgba(0,209,178,0.18);
      color: #ffffff;
      border-color: rgba(0,209,178,0.35);
    }

    @media (max-width: 900px) {
      .hero-grid { grid-template-columns: 1fr; }
      .mini-city { height: 220px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Training clustering model...")
def load_dashboard_data(k: int):
    artifacts = run_clustering(k=k)
    return (
        artifacts.data,
        artifacts.elbow_scores,
        artifacts.silhouette_scores,
        artifacts.cluster_profile,
    )


with st.sidebar:
    st.title("Parcl AI")
    st.caption("Buyer segmentation controls")
    selected_k = st.slider("Number of clusters", 2, 8, 4)

try:
    data, elbow_scores, silhouette_scores, profile = load_dashboard_data(selected_k)
except Exception as exc:
    st.error("The dashboard could not load the data or train the clustering model.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    countries = st.multiselect("Country", sorted(data["country"].dropna().unique()))
    regions = st.multiselect("Region", sorted(data["region"].dropna().unique()))
    purposes = st.multiselect("Acquisition purpose", sorted(data["acquisition_purpose"].dropna().unique()))
    client_types = st.multiselect("Client type", sorted(data["client_type"].dropna().unique()))
    segments = st.multiselect("Buyer segment", sorted(data["segment"].dropna().unique()))

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

st.markdown(
    """
    <div class="hero">
      <div class="hero-grid">
        <div>
          <div class="badge">Machine Learning Real Estate Intelligence</div>
          <div class="hero-title">3D Buyer Segmentation <span>Dashboard</span></div>
          <div class="hero-text">
            Discover hidden buyer groups, investor behavior, financing patterns, and geographic
            opportunities using K-Means clustering and hierarchical validation.
          </div>
        </div>
        <div class="mini-city">
          <div class="bar one"></div>
          <div class="bar two"></div>
          <div class="bar three"></div>
          <div class="bar four"></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No records match the selected filters. Remove one or more filters to continue.")
    st.stop()

sales_total = filtered["total_spend"].sum()
investment_rate = filtered["investment_flag"].mean() * 100
loan_rate = filtered["loan_flag"].mean() * 100
avg_satisfaction = filtered["satisfaction_score"].mean()

st.write("")
metric_cols = st.columns(4)
metrics = [
    ("Clients", f"{len(filtered):,}", "Total filtered buyers"),
    ("Linked Sales", f"${sales_total:,.0f}", "Sold property value"),
    ("Investors", f"{investment_rate:.1f}%", "Investment purpose"),
    ("Loan Applied", f"{loan_rate:.1f}%", f"Satisfaction {avg_satisfaction:.2f}/5"),
]

for col, (label, value, note) in zip(metric_cols, metrics):
    col.markdown(
        f"""
        <div class="metric-box">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tabs = st.tabs(
    [
        "Overview",
        "Investor Behavior",
        "Geographic Analysis",
        "Segment Insights",
        "Model Evaluation",
    ]
)

with tabs[0]:
    st.markdown('<div class="section-title">Buyer Segmentation Overview</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    segment_counts = (
        filtered.groupby("segment", as_index=False)
        .agg(clients=("client_id", "count"), avg_spend=("total_spend", "mean"))
        .sort_values("clients", ascending=False)
    )
    fig = px.bar(
        segment_counts,
        x="segment",
        y="clients",
        color="segment",
        color_discrete_sequence=COLORS,
        title="Cluster Distribution",
        labels={"segment": "", "clients": "Clients"},
    )
    left.plotly_chart(style_chart(fig), use_container_width=True)

    fig = px.scatter_3d(
        filtered,
        x="age",
        y="total_spend",
        z="satisfaction_score",
        color="segment",
        size="property_count",
        color_discrete_sequence=COLORS,
        hover_data=["client_id", "country", "region", "acquisition_purpose"],
        title="3D Buyer Positioning",
        labels={"age": "Age", "total_spend": "Total spend", "satisfaction_score": "Satisfaction"},
    )
    fig.update_layout(
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#d8e2ef", gridcolor="rgba(255,255,255,0.12)"),
            yaxis=dict(color="#d8e2ef", gridcolor="rgba(255,255,255,0.12)"),
            zaxis=dict(color="#d8e2ef", gridcolor="rgba(255,255,255,0.12)"),
        )
    )
    right.plotly_chart(style_chart(fig), use_container_width=True)

with tabs[1]:
    st.markdown('<div class="section-title">Investor Behavior Dashboard</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    purpose = filtered.groupby(["segment", "acquisition_purpose"], as_index=False).size()
    purpose = purpose.rename(columns={"size": "clients"})
    fig = px.bar(
        purpose,
        x="segment",
        y="clients",
        color="acquisition_purpose",
        barmode="group",
        color_discrete_sequence=COLORS,
        title="Investment vs Personal Use",
    )
    c1.plotly_chart(style_chart(fig), use_container_width=True)

    loan = filtered.groupby(["segment", "loan_applied"], as_index=False).size()
    loan = loan.rename(columns={"size": "clients"})
    fig = px.bar(
        loan,
        x="segment",
        y="clients",
        color="loan_applied",
        barmode="stack",
        color_discrete_sequence=["#64748b", "#00d1b2"],
        title="Loan Behavior by Segment",
    )
    c2.plotly_chart(style_chart(fig), use_container_width=True)

    spend = filtered.groupby("segment", as_index=False).agg(
        avg_sale_price=("avg_sale_price", "mean"),
        avg_area=("avg_floor_area_sqft", "mean"),
    )
    fig = px.scatter(
        spend,
        x="avg_area",
        y="avg_sale_price",
        color="segment",
        size="avg_sale_price",
        color_discrete_sequence=COLORS,
        title="Average Unit Area vs Sale Price",
        labels={"avg_area": "Average floor area", "avg_sale_price": "Average sale price"},
    )
    st.plotly_chart(style_chart(fig, height=380), use_container_width=True)

with tabs[2]:
    st.markdown('<div class="section-title">Geographic Buyer Analysis</div>', unsafe_allow_html=True)
    geo = filtered.groupby(["country", "region", "segment"], as_index=False).agg(
        clients=("client_id", "count"),
        sales=("total_spend", "sum"),
    )
    g1, g2 = st.columns(2)
    fig = px.treemap(
        geo,
        path=["country", "region", "segment"],
        values="clients",
        color="segment",
        color_discrete_sequence=COLORS,
        title="Buyer Segments by Country and Region",
    )
    g1.plotly_chart(style_chart(fig), use_container_width=True)

    top_regions = geo.groupby("region", as_index=False)["clients"].sum()
    top_regions = top_regions.sort_values("clients", ascending=False).head(12)
    fig = px.bar(
        top_regions,
        x="clients",
        y="region",
        orientation="h",
        color="clients",
        color_continuous_scale=["#0b1220", "#00d1b2"],
        title="Top Buyer Regions",
    )
    g2.plotly_chart(style_chart(fig), use_container_width=True)

with tabs[3]:
    st.markdown('<div class="section-title">Segment Insights Panel</div>', unsafe_allow_html=True)
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
    for i, row in filtered_profile.reset_index(drop=True).iterrows():
        insight_cols[i % 2].markdown(
            f"""
            <div class="insight">
              <b>{row['segment']}</b>
              <p>
                {int(row['clients'])} clients | Avg age {row['avg_age']:.1f} | Satisfaction {row['avg_satisfaction']:.2f}<br>
                Investment share {row['investment_rate'] * 100:.1f}% | Loan share {row['loan_rate'] * 100:.1f}%<br>
                Avg linked spend ${row['avg_total_spend']:,.0f} | Core region {row['top_region']}
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")
    st.dataframe(filtered_profile, use_container_width=True, hide_index=True)

with tabs[4]:
    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    fig = px.line(elbow_scores, x="k", y="inertia", markers=True, title="Elbow Method")
    m1.plotly_chart(style_chart(fig), use_container_width=True)

    fig = px.line(silhouette_scores, x="k", y="silhouette_score", markers=True, title="Silhouette Score")
    m2.plotly_chart(style_chart(fig), use_container_width=True)
    st.dataframe(profile, use_container_width=True, hide_index=True)
