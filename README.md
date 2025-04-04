# Credit Card Customer Segmentation using K-Means Clustering

**Project Overview:**

This project aims to segment credit card customers into distinct groups based on their spending behavior and financial attributes. Customer segmentation is valuable for businesses to understand their customer base and tailor marketing strategies accordingly.

**Project Goals and Objectives:**

- **Goal:** To identify meaningful customer segments within a credit card dataset.
- **Objectives:**
    - Preprocess and clean the data.
    - Apply K-Means clustering to group customers.
    - Evaluate the clustering results.
    - Visualize the segments in a reduced-dimensional space.

**Techniques and Technologies Used:**

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Clustering Algorithm:** K-Means
- **Dimensionality Reduction:** Principal Component Analysis (PCA)
- **Evaluation Metrics:** Silhouette Score, Inertia

**Code Structure and Logic:**

1. **Data Loading and Preprocessing:**
    - Import necessary libraries (Pandas, NumPy, etc.).
    - Load the credit card dataset using Pandas.
    - Handle missing values by imputation.
    - Standardize numerical features using `StandardScaler`.

2. **Dimensionality Reduction (PCA):**
    - Apply PCA to reduce the dataset's dimensionality while preserving most of the variance.
    - This helps in visualization and improves clustering performance.

3. **K-Means Clustering:**
    - Initialize a K-Means model with a chosen number of clusters (e.g., 3).
    - Fit the model to the scaled data to identify clusters.
    - Assign cluster labels to each data point.

4. **Cluster Evaluation:**
    - **Elbow Method:** Analyze the Within-Cluster Sum of Squares (WCSS) for different numbers of clusters to find the optimal 'K' value.
    - **Silhouette Score:** Evaluate the quality of clusters using the silhouette score, which measures how similar a data point is to its own cluster compared to other clusters.

5. **Cluster Visualization:**
    - Visualize the clusters using scatter plots in the reduced-dimensional PCA space.
    - Color-code data points based on their cluster assignments.
  
   
   ![image](https://github.com/user-attachments/assets/db04b030-cd9e-4ea6-8b1e-639ea9fdf006)



**Algorithms:**

- **K-Means:** An iterative clustering algorithm that partitions data into clusters based on the distance between data points and cluster centroids.
- **PCA:** A dimensionality reduction technique that finds principal components (linear combinations of features) that capture the most variance in the data.

**Conclusions:**

- The project successfully segmented credit card customers into distinct groups.
- The optimal number of clusters was determined using the elbow method and silhouette analysis.
- Visualization using PCA helps in understanding the characteristics of each cluster.

**Installation and Usage:**

1. Install the required libraries:
bash pip install pandas numpy scikit-learn matplotlib seaborn
2. Run the code in a Python environment (e.g., Google Colab).
3. Modify the input data file path as needed.
4. Experiment with different parameter settings for K-Means and PCA.
