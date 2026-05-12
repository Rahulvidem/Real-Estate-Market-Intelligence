# Parcl Buyer Segmentation and Investment Profiling

This project applies machine learning based clustering to Parcl real estate buyer data. It cleans client and property records, encodes buyer attributes, scales numeric behavior, runs K-Means and hierarchical clustering, and presents buyer intelligence in a Streamlit dashboard.

## Project Structure

```text
.
+-- app.py
+-- data/
|   +-- clients.csv
|   +-- properties.csv
+-- docs/
|   +-- research_paper.md
+-- requirements.txt
+-- src/
    +-- pipeline.py
```

## Setup

Install Python 3.10 or newer, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Methodology

1. Data cleaning: duplicate removal, missing value handling, date parsing, label normalization, and sale price conversion.
2. Feature engineering: age, loan flag, investment flag, company flag, purchase count, total spend, average sale price, and average floor area.
3. Encoding: one-hot encoding for client type, gender, country, region, acquisition purpose, and referral channel.
4. Scaling: StandardScaler for numeric variables.
5. Clustering: K-Means as the primary segmentation model and hierarchical clustering as a validation lens.
6. Evaluation: elbow method and silhouette score across cluster counts from 2 to 8.
7. Interpretation: segment profiles are built from investment purpose, geography, financing behavior, demographics, spend, and satisfaction.

## Dashboard Pages

- Buyer Segmentation Overview
- Investor Behavior Dashboard
- Geographic Buyer Analysis
- Segment Insights Panel
- Model Evaluation

## Dataset Notes

The project includes 2,000 client records and 10,000 property records. Property rows marked `Available` do not have a buyer reference, so investment behavior is calculated from sold, client-linked properties.
