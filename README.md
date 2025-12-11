# **Market Basket Analysis with Customer Segmentation**

## **Project Overview**
This project develops an intelligent retail analytics system that integrates customer segmentation with association rule mining. Instead of generating generic market basket rules, we create a **two-tier analytics pipeline** that first identifies distinct customer segments through clustering, then discovers segment-specific purchasing patterns. The system provides actionable insights for targeted marketing, personalized recommendations, and inventory optimization tailored to different customer groups.

## **Dataset Choice**
**Selected:** Online Retail II Dataset from Kaggle

**Why this dataset:**
1. **Real-World Scale:** ~1M+ transactions from a UK-based online retailer (01/12/2009 - 09/12/2011)
2. **Rich Features:** Customer IDs, product details, quantities, timestamps, and geographic data
3. **Perfect Structure:** Transaction data ideal for both clustering (customer behavior) and ARM (product associations)
4. **Realistic Challenges:** Missing values, cancellations, returns - providing authentic data cleaning opportunities

**Dataset Statistics:**
- InvoiceNo: Transaction identifier (with 'C' prefix for cancellations)
- StockCode: Product identifier
- Description: Product name
- Quantity: Purchase quantity
- InvoiceDate: Transaction timestamp
- UnitPrice: Price per unit in £
- CustomerID: Unique customer identifier
- Country: Customer location

## **Architecture Sketch**

### **Core Design Principles:**
1. **Two-Phase Pipeline:** Separate but integrated clustering and association mining
2. **RFM-Driven Segmentation:** Use Recency, Frequency, Monetary metrics for customer profiling
3. **Segment-Specific Analysis:** Different association rules for different customer types
4. **Visual Integration:** Bridge customer segments with their purchasing patterns

```
Raw Transaction Data
    ↓
Data Cleaning Pipeline
    ├── Handle missing CustomerIDs
    ├── Filter cancellations/returns
    ├── Remove outliers
    ├── Standardize descriptions
    ↓
Feature Engineering
    ├── RFM Calculation (Recency, Frequency, Monetary)
    ├── Product category features
    ├── Behavioral metrics
    ↓
Customer Segmentation (Clustering Layer)
    ├── K-means / DBSCAN on RFM features
    ├── Hierarchical clustering validation
    ├── Segment profile creation
    ↓
Segment-Specific Transaction Preparation
    ↓
Association Rule Mining (Per Segment)
    ├── Apriori algorithm application
    ├── FP-Growth comparison
    ├── Rule quality metrics (support, confidence, lift)
    ↓
Cross-Segment Analysis & Visualization
    ├── Compare rules across segments
    ├── Identify unique vs. common patterns
    └── Generate business insights
```

**Outputs:**
- Customer segments with behavioral profiles
- Segment-specific association rules (product affinities)
- Comparison: Global rules vs. segment-specific rules
- Visualization: Cluster plots + rule networks + segment characteristics