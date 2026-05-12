from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class ClusterArtifacts:
    data: pd.DataFrame
    feature_matrix: np.ndarray
    elbow_scores: pd.DataFrame
    silhouette_scores: pd.DataFrame
    cluster_profile: pd.DataFrame
    preprocessor: ColumnTransformer
    kmeans: KMeans


NUMERIC_FEATURES = [
    "age",
    "satisfaction_score",
    "loan_flag",
    "property_count",
    "total_spend",
    "avg_sale_price",
    "avg_floor_area_sqft",
    "investment_flag",
    "company_flag",
]

CATEGORICAL_FEATURES = [
    "client_type",
    "gender",
    "country",
    "region",
    "acquisition_purpose",
    "referral_channel",
]


def load_raw_data(
    clients_path: str | Path = DATA_DIR / "clients.csv",
    properties_path: str | Path = DATA_DIR / "properties.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clients = pd.read_csv(clients_path)
    properties = pd.read_csv(properties_path)
    return clients, properties


def _clean_money(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace({"nan": np.nan, "": np.nan})
        .astype(float)
    )


def clean_and_merge(clients: pd.DataFrame, properties: pd.DataFrame) -> pd.DataFrame:
    clients = clients.copy()
    properties = properties.copy()

    clients.columns = clients.columns.str.strip().str.lower()
    properties.columns = properties.columns.str.strip().str.lower()

    clients = clients.drop_duplicates(subset=["client_id"]).reset_index(drop=True)

    text_columns = [
        "client_type",
        "gender",
        "country",
        "region",
        "acquisition_purpose",
        "loan_applied",
        "referral_channel",
    ]
    for col in text_columns:
        clients[col] = clients[col].fillna("Unknown").astype(str).str.strip()

    replacement_map = {
        "Company": "Corporate",
        "Home": "Personal Use",
        "Personal": "Personal Use",
    }
    clients["client_type"] = clients["client_type"].replace(replacement_map)
    clients["acquisition_purpose"] = clients["acquisition_purpose"].replace(replacement_map)
    clients["satisfaction_score"] = pd.to_numeric(
        clients["satisfaction_score"], errors="coerce"
    ).fillna(clients["satisfaction_score"].median())
    clients["date_of_birth"] = pd.to_datetime(
        clients["date_of_birth"], errors="coerce", format="mixed"
    )
    today = pd.Timestamp.today().normalize()
    clients["age"] = ((today - clients["date_of_birth"]).dt.days / 365.25).round(1)
    clients["age"] = clients["age"].fillna(clients["age"].median())
    clients["loan_flag"] = clients["loan_applied"].str.lower().eq("yes").astype(int)
    clients["investment_flag"] = (
        clients["acquisition_purpose"].str.lower().eq("investment").astype(int)
    )
    clients["company_flag"] = clients["client_type"].str.lower().eq("corporate").astype(int)

    properties["sale_price"] = _clean_money(properties["sale_price"])
    properties["floor_area_sqft"] = pd.to_numeric(
        properties["floor_area_sqft"], errors="coerce"
    )
    properties["transaction_date"] = pd.to_datetime(
        properties["transaction_date"], errors="coerce", format="mixed"
    )
    sold = properties[
        properties["listing_status"].astype(str).str.lower().eq("sold")
        & properties["client_ref"].notna()
        & properties["client_ref"].astype(str).str.len().gt(0)
    ].copy()

    property_summary = (
        sold.groupby("client_ref")
        .agg(
            property_count=("listing_id", "count"),
            total_spend=("sale_price", "sum"),
            avg_sale_price=("sale_price", "mean"),
            avg_floor_area_sqft=("floor_area_sqft", "mean"),
            first_purchase=("transaction_date", "min"),
            last_purchase=("transaction_date", "max"),
        )
        .reset_index()
        .rename(columns={"client_ref": "client_id"})
    )

    data = clients.merge(property_summary, on="client_id", how="left")
    fill_zero = ["property_count", "total_spend", "avg_sale_price", "avg_floor_area_sqft"]
    data[fill_zero] = data[fill_zero].fillna(0)
    data["buyer_status"] = np.where(data["property_count"] > 0, "Converted", "Lead")
    return data


def make_preprocessor() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", encoder, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def score_cluster_range(feature_matrix: np.ndarray, k_min: int = 2, k_max: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    elbow_rows: list[dict[str, float]] = []
    silhouette_rows: list[dict[str, float]] = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(feature_matrix)
        elbow_rows.append({"k": k, "inertia": float(model.inertia_)})
        silhouette_rows.append({"k": k, "silhouette_score": float(silhouette_score(feature_matrix, labels))})
    return pd.DataFrame(elbow_rows), pd.DataFrame(silhouette_rows)


def choose_optimal_k(silhouette_scores: pd.DataFrame, default: int = 4) -> int:
    if silhouette_scores.empty:
        return default
    best = silhouette_scores.sort_values("silhouette_score", ascending=False).iloc[0]
    return int(best["k"])


def assign_segment_names(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    profiles = (
        data.groupby("cluster")
        .agg(
            avg_age=("age", "mean"),
            investment_rate=("investment_flag", "mean"),
            loan_rate=("loan_flag", "mean"),
            corporate_rate=("company_flag", "mean"),
            avg_spend=("total_spend", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            buyers=("client_id", "count"),
        )
        .sort_index()
    )

    remaining = set(profiles.index)
    names: dict[int, str] = {}

    if remaining:
        c = profiles.loc[list(remaining), "corporate_rate"].idxmax()
        names[int(c)] = "Corporate Buyers"
        remaining.remove(c)
    if remaining:
        c = profiles.loc[list(remaining), "loan_rate"].idxmax()
        names[int(c)] = "First-Time Buyers"
        remaining.remove(c)
    if remaining:
        c = profiles.loc[list(remaining), "avg_spend"].idxmax()
        names[int(c)] = "Luxury Investors"
        remaining.remove(c)
    if remaining:
        c = profiles.loc[list(remaining), "investment_rate"].idxmax()
        names[int(c)] = "Global Investors"
        remaining.remove(c)

    fallback = ["Value Home Buyers", "Digital Leads", "Regional Investors", "Mixed Buyers"]
    for i, c in enumerate(sorted(remaining)):
        names[int(c)] = fallback[i % len(fallback)]

    data["segment"] = data["cluster"].map(names)
    return data


def build_cluster_profile(data: pd.DataFrame) -> pd.DataFrame:
    profile = (
        data.groupby(["cluster", "segment"])
        .agg(
            clients=("client_id", "count"),
            avg_age=("age", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            investment_rate=("investment_flag", "mean"),
            loan_rate=("loan_flag", "mean"),
            corporate_rate=("company_flag", "mean"),
            avg_total_spend=("total_spend", "mean"),
            avg_property_count=("property_count", "mean"),
            top_country=("country", lambda x: x.mode().iat[0] if not x.mode().empty else "Unknown"),
            top_region=("region", lambda x: x.mode().iat[0] if not x.mode().empty else "Unknown"),
            top_channel=("referral_channel", lambda x: x.mode().iat[0] if not x.mode().empty else "Unknown"),
        )
        .reset_index()
    )
    pct_columns = ["investment_rate", "loan_rate", "corporate_rate"]
    for col in pct_columns:
        profile[col] = (profile[col] * 100).round(1)
    money_columns = ["avg_total_spend"]
    for col in money_columns:
        profile[col] = profile[col].round(2)
    profile["avg_age"] = profile["avg_age"].round(1)
    profile["avg_satisfaction"] = profile["avg_satisfaction"].round(2)
    profile["avg_property_count"] = profile["avg_property_count"].round(2)
    return profile.sort_values("clients", ascending=False)


def run_clustering(k: int | None = 4) -> ClusterArtifacts:
    clients, properties = load_raw_data()
    data = clean_and_merge(clients, properties)
    preprocessor = make_preprocessor()
    feature_matrix = preprocessor.fit_transform(data[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    elbow, silhouettes = score_cluster_range(feature_matrix)
    selected_k = k or choose_optimal_k(silhouettes)

    kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=25)
    data["cluster"] = kmeans.fit_predict(feature_matrix)
    data = assign_segment_names(data)

    hierarchical = AgglomerativeClustering(n_clusters=selected_k)
    data["hierarchical_cluster"] = hierarchical.fit_predict(feature_matrix)
    profile = build_cluster_profile(data)

    return ClusterArtifacts(
        data=data,
        feature_matrix=feature_matrix,
        elbow_scores=elbow,
        silhouette_scores=silhouettes,
        cluster_profile=profile,
        preprocessor=preprocessor,
        kmeans=kmeans,
    )
