# CSC172 Data Mining Project Proposal
**Student:** Emmanuel Fitz C. Ciano
**Date:** December 11, 2025

## 1. Project Title
Smart Retail Analytics: Customer Segmentation-Driven Association Rule Mining

## 2. Problem Statement
Traditional market basket analysis treats all customers equally, generating generic association rules that fail to capture diverse purchasing behaviors. Retailers struggle to personalize recommendations because they lack insights into how different customer segments purchase products together. By treating all transactions uniformly, businesses miss opportunities for targeted marketing, personalized promotions, and inventory optimization specific to customer groups. There's a need to integrate customer segmentation with association rule mining to discover segment-specific purchasing patterns that reflect real consumer behavior diversity.

## 3. Objectives
- Develop a hybrid analytics pipeline combining clustering and association rule mining
- Segment customers using RFM (Recency, Frequency, Monetary) analysis and purchase behavior
- Discover distinct association rules specific to each customer segment
- Compare segment-specific patterns against global market basket analysis
- Provide actionable insights for targeted marketing and inventory management
- Create visualizations that bridge customer segmentation with product affinities

## 4. Dataset Plan
- **Primary Dataset:** Online Retail II Dataset (Kaggle)
  - ~1M+ transactions from UK-based online retailer (2009-2011)
  - Contains customer IDs, product details, quantities, timestamps, and geographic data
  - Real-world challenges: missing values, cancellations, returns
- **Domain:** Retail analytics, e-commerce, consumer behavior
- **Size:** Sufficient for meaningful clustering (1000+ customers after cleaning)

## 5. Technical Approach
- **Clustering Algorithms:** K-means, DBSCAN, Hierarchical clustering for customer segmentation
- **Association Mining:** Apriori algorithm, FP-Growth for market basket analysis
- **Feature Engineering:** RFM metrics, product category features, behavioral patterns
- **Framework:** Python with scikit-learn, mlxtend, pandas, matplotlib
- **Evaluation:** Silhouette score for clustering; support, confidence, lift for association rules
- **Visualization:** Cluster plots, rule networks, comparative dashboards

## 6. Expected Challenges & Mitigations
- **Challenge:** Data cleaning complexity (missing CustomerIDs, cancellations, outliers)
  - **Solution:** Systematic cleaning pipeline with validation checks

- **Challenge:** Determining optimal number of customer segments
  - **Solution:** Multiple validation methods (elbow method, silhouette score, dendrogram)

- **Challenge:** Computational complexity with large transaction data
  - **Solution:** Sampling strategies, efficient data structures, FP-Growth algorithm

- **Challenge:** Interpreting and validating segment-specific rules
  - **Solution:** Business context integration, cross-validation with domain knowledge