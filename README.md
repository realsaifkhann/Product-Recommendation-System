# Product Recommendation System

A collaborative filtering-based recommendation system built using K-Means clustering, deployed as an interactive web application with Streamlit.

---

## Overview

E-commerce platforms rely on recommendation systems to suggest relevant products to users. This project implements **user-based collaborative filtering** — grouping users with similar rating patterns and predicting what a user might like based on their cluster's behaviour.

---

## Dataset

| Column | Description |
|---|---|
| `userid` | Unique identifier for each user |
| `productid` | Unique identifier for each product |
| `rating` | Star rating given by the user (1–5) |
| `date` | Timestamp of the rating (dropped during preprocessing) |

- Total records: **78,245**
- After filtering: **583 rows · 515 users · 246 products**
- Matrix sparsity: **~99%** (typical for real-world recommendation datasets)

---

## Workflow

**Phase 1 — Data Preprocessing**
- Dropped the `date` column (not required for collaborative filtering)
- Removed duplicate entries
- Filtered out users and products with fewer than 2 interactions
- Built a User-Item Matrix (users as rows, products as columns, ratings as values)
- Filled missing values with each user's mean rating
- Confirmed high sparsity (~99%)

**Phase 2 — Model Building**
- Applied StandardScaler to normalize the user-item matrix
- Trained K-Means clustering to group similar users
- Used Silhouette Score to tune the number of clusters (k=2 to k=6)
- Selected k=4 as the optimal number of clusters
- Evaluated Hierarchical Clustering using a dendrogram for comparison

**Phase 3 — Evaluation**
- Built a prediction function using cluster-average ratings
- Evaluated on an 80/20 train-test split using RMSE

---

## Results

| Model | Silhouette Score | RMSE |
|---|---|---|
| K-Means (k=4) | ~0.90 | ~0.31 |
| Hierarchical (n=2) | ~0.75 | N/A |

K-Means was selected as the final model due to its higher silhouette score, lower RMSE, and better scalability.

---

## Deployment

Built an Amazon-style interactive web app using Streamlit with the following features:

- Select any customer ID to get personalized product recommendations
- Displays predicted star rating and match percentage per product
- Shows user cluster, similar users count, and model metrics
- Customers Also Viewed section for additional suggestions

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Streamlit

---

## Cold Start Handling

New users with no rating history cannot be assigned to a cluster. This is known as the **Cold Start Problem**. The fallback strategy is to recommend globally top-rated products until sufficient user interaction data is collected.

---
