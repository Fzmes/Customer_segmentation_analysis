# Customer segmentation analysis
---

## 📊 Dataset Description

**File**: `E-commerce Customer behavior.csv`  
**Contents**:
- Customer ID
- Geographic and demographic info
- Spending behavior
- Interaction with the platform

The dataset is used as the input for clustering to uncover meaningful customer segments.

---

## 🧠 Clustering Techniques Used

1. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**  
   - Captures clusters of varying shapes
   - Good at detecting outliers

2. **Hierarchical Clustering (Agglomerative)**  
   - Builds a dendrogram to represent nested clusters
   - No need to pre-specify number of clusters

3. **Mean Shift Clustering**  
   - Centroid-based algorithm that does not require the number of clusters in advance
   - Results visualized in `mean_shift_result.png`

