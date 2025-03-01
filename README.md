Customer Segmentation using K-Means Clustering

Project Overview

This project applies unsupervised machine learning techniques to segment customers based on their purchasing behavior. Using K-Means Clustering, Hierarchical Clustering, and DBSCAN, we analyze transactional data to derive meaningful customer groups, aiding businesses in targeted marketing and customer retention strategies.

Tech Stack

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

Machine Learning Algorithms: K-Means, Hierarchical Clustering, DBSCAN

Evaluation Metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Gap Statistic

Methodology

Data Collection & Preprocessing:

Cleaned missing values, removed duplicates, and filtered invalid transactions.

Converted transaction timestamps to datetime format for recency analysis.

Feature Engineering:

Created RFM (Recency, Frequency, Monetary) Analysis features.

Applied Principal Component Analysis (PCA) for dimensionality reduction.

Clustering Models:

K-Means: Used the Elbow Method & Silhouette Score to determine optimal K.

Hierarchical Clustering: Built dendrograms for comparison.

DBSCAN: Identified potential outliers and anomalies.

Model Evaluation:

Compared clustering models using Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, and Gap Statistic.

Optimized K-Means hyperparameters (k-means++, max_iter=500, tol=1e-6).

Key Findings & Business Impact

Segmented customers into three groups: Regular Buyers, VIP High-Spenders, and Loyal Frequent Buyers.

K-Means (K=3) was the best clustering model, balancing compactness and separation.

Hierarchical Clustering provided insights into customer evolution over time.

DBSCAN helped detect outliers (potential fraud or VIP customers).

Suggested marketing strategies: Personalized offers for high-value customers, loyalty programs for frequent buyers, and discounts for low-engagement customers.

How to Run This Project

Clone the repository:

git clone https://github.com/aftabshaikhraza/customer-segmentation-ml.git
cd customer-segmentation-ml

Install dependencies:

pip install -r requirements.txt

Run the Jupyter Notebook:

jupyter notebook Customer_Segmentation.ipynb

##Dataset Information
The dataset used in this project is sourced from **UC Ervine Machine learning Repository**.  
It contains **customer transaction records**, which were preprocessed for analysis.  

**Dataset Link:** https://archive.ics.uci.edu/dataset/352/online+retail  
**Dataset citation:** Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33.
**Dataset File:** `dataset.xlsx`

Future Improvements

Test additional clustering techniques (Gaussian Mixture Models, Agglomerative Clustering).

Expand feature engineering with time-based analysis of customer behavior.

Apply deep learning-based customer segmentation techniques.

Contact & Contribution

Contributions are welcome. Feel free to fork the repository, submit pull requests, or reach out for collaborations.

