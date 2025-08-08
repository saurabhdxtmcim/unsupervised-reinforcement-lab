# Unsupervised & Reinforcement Learning Lab

**Python • TensorFlow • Gymnasium**  
**License:** MIT

A comprehensive collection of machine learning implementations focusing on **unsupervised learning algorithms** and **reinforcement learning techniques**.  
This repository contains hands-on Python code with detailed mathematical explanations and practical applications.

---

## 📚 Table of Contents
- 🧠 Project Overview  
- 📁 Project Structure  
- 🛠️ Technical Stack  
- 🚀 Getting Started  
- 🔬 Algorithm Implementations  
- 📈 Features  
- 🤝 Contributing  
- 📄 License  
- 🎯 Learning Objectives  

---

## 🧠 Project Overview
This lab explores advanced machine learning techniques beyond supervised learning, featuring:

- **Unsupervised Learning:** Clustering, dimensionality reduction, and anomaly detection  
- **Reinforcement Learning:** Deep Q-Learning implementations with a focus on education and clarity  
- **Recommender Systems:** Collaborative filtering and content-based approaches  
- **Interactive Visualizations:** Bokeh and Plotly-powered data exploration tools  

---

## 📁 Project Structure

### 🔍 Unsupervised Learning Algorithms  
**Clustering & Pattern Recognition**  
- `k_means_clustering.py` – Complete K-means implementation with cluster assignment and centroid optimization  
- `anomaly_detection.py` – Gaussian model-based anomaly detection for server behavior monitoring  

**Dimensionality Reduction**  
- `principal_component_analysis.py` – Interactive PCA with Bokeh visualizations  
- `principal_component_analysis_visual.py` – Visual PCA demonstrations with Matplotlib  
- `helpers/pca_utils.py` – Utility functions for PCA visualization and 3D/2D plotting  

---

### 🎮 Reinforcement Learning  
**Deep Q-Learning**  
- `deep_q_learning_explained.py` – Educational DQN implementation with detailed explanations and mathematical foundations  

---

### 🎬 Recommender Systems  
**Collaborative Filtering**  
- `collaborative_recommender.py` – Matrix factorization-based movie recommendation system  
- `movie_recommender_training.py` – Training pipeline for collaborative filtering models  

**Content-Based Filtering**  
- `content_filtering_neural_network.py` – Neural network approach to content-based recommendations  
- `deep_learning_content_filtering.py` – Deep learning implementation for content filtering  
- `helpers/recsysNN_utils.py` – Neural network utilities for recommendation systems  

---

### 📊 Data & Resources  
- `data/` – Movie ratings datasets, user preferences, and training data  

Generated Outputs:  
- `pca_dimensionality_reduction.png` – PCA visualization results  
- `simple_pca_example.png` – Basic PCA demonstration  
- `principal_component_analysis.html` – Interactive PCA dashboard  

---

## 🛠️ Technical Stack
- **Core ML:** TensorFlow 2.19, Keras, Scikit-learn  
- **Data Processing:** NumPy, Pandas  
- **Visualization:** Matplotlib, Bokeh, Plotly  
- **Future RL Extensions:** Prepared for Gymnasium environments  
- **Development:** Python 3.11+, Jupyter-compatible  

---

## 🚀 Getting Started

### Prerequisites  
- Python 3.11 or higher  
- Git  

### Installation  
```bash
git clone https://github.com/yourusername/unsupervised-reinforcement-lab.git
cd unsupervised-reinforcement-lab
```

Create a virtual environment:  
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install dependencies:  
```bash
pip install -r requirements.txt
```

---

## Quick Start Examples  
Run K-means Clustering:  
```bash
python k_means_clustering.py
```

Explore Deep Q-Learning Concepts:  
```bash
python deep_q_learning_explained.py
```

Explore PCA Visualization:  
```bash
python principal_component_analysis.py
```

Test Movie Recommender:  
```bash
python collaborative_recommender.py
```

---

## 🔬 Algorithm Implementations  

### Unsupervised Learning  
- **K-means Clustering:** Iterative centroid optimization with cluster assignment  
- **Principal Component Analysis:** Dimensionality reduction with interactive visualization  
- **Anomaly Detection:** Gaussian distribution modeling for outlier detection  

### Reinforcement Learning  
- **Deep Q-Networks (DQN):** Educational example with detailed explanations and mathematical foundations  

### Recommender Systems  
- **Collaborative Filtering:** Matrix factorization with user-item interactions  
- **Content-Based Filtering:** Feature-based recommendation using neural networks  
- **Hybrid Approaches:** Combining collaborative and content-based methods  

---

## 📈 Features  
✅ Mathematical Foundations: Detailed algorithm explanations with mathematical notation  
✅ Interactive Visualizations: Real-time plots and 3D visualizations  
✅ Modular Design: Reusable utility functions and helper modules  
✅ Performance Tracking: Training progress monitoring and metrics  
✅ Cross-Platform: Windows, macOS, and Linux compatibility  

---

## 🤝 Contributing  
This is a personal learning repository, but suggestions and improvements are welcome! Feel free to:  
- Report bugs or issues  
- Suggest new algorithms to implement  
- Improve documentation or code clarity  

---

## 📄 License  
This project is licensed under the MIT License – see the LICENSE file for details.  

---

## 🎯 Learning Objectives  
This repository serves as a practical exploration of:  
- Advanced machine learning techniques beyond supervised learning  
- Mathematical foundations of ML algorithms  
- Practical implementation challenges and solutions  
- Visualization techniques for complex data  
- Reinforcement learning in simulated environments  

---

Happy Learning! 🚀
