# ARC 2024

### Introduction

This is a project using the ARC 2024 competition dataset aiming to build an algorithm that is capable of solving novel abstract reasoning tasks that it has never seen before. This is the crux of artificial general intelligence (AGI) and is a component of AI systems that can learn new skills and solve open-ended systems, rather than AI systems that simply memorize data. The dataset's symbolic nature makes it infeasible to use traditional feature engineering, motivating the use of unsupervised learning. The analysis applies dimensionality reduction methods (PCA, t-SNE, UMAP, Spectral Embedding, etc.) to project high-dimensional representations into low-dimensional spaces, revealing interpretable and meaningful structures in the data that are not visible in the raw feature space. Clustering with KMeans and GMM applied before and after dimensionality reduction showed a significant improvement in separability (silhouette scores 0.61 to 0.82), suggesting that non-linear feature transformation is critical in extracting meaningful structure from abstract task spaces. These findings demonstrate the potential of unsupervised learning techniques for interpreting abstract reasoning tasks, which may support downstream applications in transfer learning or meta-learning.

### Unsupervised Learning

This notebook focuses on unsupervised learning. It uses dimensionality reduction and non-linear manifold learning techniques on the dataset, then performs clustering, to prepare the dataset for further analysis. It also trains a neural network on it as a baseline.

### Dataset
The ARC dataset can be downloaded from the Kaggle competition page. Ensure you have the dataset downloaded and placed in the appropriate directory within the project. The link is here: https://www.kaggle.com/competitions/arc-prize-2024/data. 

### Running the Code

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open the Notebook:
Navigate to the notebook file 'arc-ml-a3-v2.ipynb' in the Jupyter Notebook interface and open it.

4. Run the Notebook:

Follow these steps within the notebook:

- **Data Loading:** Ensure the ARC dataset is loaded correctly.
- **Data Preprocessing:** Follow the preprocessing steps to flatten nested lists (arrays) and perform any necessary transformations.
- **Clustering:** Apply K-Means and Expectation Maximization (EM) clustering algorithms to the dataset.
- **Dimensionality Reduction:** Apply PCA (Principal Component Analysis), ICA (Independent Component Analysis), and RP (Random Projection) to transform the dataset.
- **Non-linear Manifold Learning:** Use t-SNE (t-Distributed Stochastic Neighbor Embedding), MDS (Multidimensional Scaling), Spectral Embedding, and UMAP (Uniform Manifold Approximation and Projection) for comparison and visualization.
- **Re-apply Clustering:** Run clustering algorithms on the transformed datasets to evaluate the impact of these transformations on clustering performance.
- **Neural Network Models:** Train neural network models on the transformed datasets, using clusters as new features, and evaluate the performance using accuracy, learning curves, and training time.
- **Evaluation:** Use the Silhouette Score to evaluate clustering performance, considering cohesion within clusters and separation between clusters.

For more information on manifold learning on scikit-learn, see the section on their website: https://scikit-learn.org/stable/modules/manifold.html.

### Results
Results will be displayed within the Jupyter Notebook, including:

- Clustering results visualized for both original and transformed datasets.
- Silhouette Scores for different clustering and dimensionality reduction techniques.
- Neural network performance metrics.

### Future Work

Further research may include:

- Exploring advanced clustering methods like hierarchical clustering and DBSCAN.
- Fine-tuning hyperparameters using Bayesian optimization.
- Incorporating ensemble methods and transformer-based embeddings.
- Investigating reinforcement learning for abstract reasoning tasks.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

