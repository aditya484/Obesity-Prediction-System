<div align="center">

# 🧬 Obesity Risk Prediction System

### *End-to-End Machine Learning · Full-Stack Web Application*

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-API-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-FF6600?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+&logoColor=white)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

[![GitHub stars](https://img.shields.io/github/stars/aditya484/Obesity-Prediction-System?style=social)](https://github.com/aditya484/Obesity-Prediction-System/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/aditya484/Obesity-Prediction-System?style=social)](https://github.com/aditya484/Obesity-Prediction-System/network/members)
[![GitHub issues](https://img.shields.io/github/issues/aditya484/Obesity-Prediction-System?color=red)](https://github.com/aditya484/Obesity-Prediction-System/issues)

<br/>

> **Predict obesity levels with AI — powered by XGBoost, Flask & Streamlit.**  
> A complete ML pipeline from raw data to a deployed web app, built for academic excellence & real-world impact. 🏥

<br/>

[🚀 Get Started](#-quick-start) · [📊 ML Pipeline](#-machine-learning-pipeline) · [🌐 Deployment](#-deployment) · [🤝 Connect](#-connect-with-me)

</div>

---

## 📌 Table of Contents

- [✨ About the Project](#-about-the-project)
- [🎯 Key Features](#-key-features)
- [🏗️ Project Structure](#️-project-structure)
- [🧠 Machine Learning Pipeline](#-machine-learning-pipeline)
- [🚀 Quick Start](#-quick-start)
- [🖥️ Running the Apps](#️-running-the-apps)
- [🌐 Deployment](#-deployment)
- [📦 Tech Stack](#-tech-stack)
- [🤝 Connect with Me](#-connect-with-me)

---

## ✨ About the Project

The **Obesity Risk Prediction System** is a production-grade, end-to-end machine learning application that predicts a person's **obesity level** based on their dietary habits, physical activity, and lifestyle choices.

This project bridges the gap between **Data Science** and **Software Engineering** by combining:
- 🔬 A rigorous **ML pipeline** with SMOTE, feature engineering & multi-model comparison
- 🌐 A **Flask full-stack web app** with a glassmorphism UI
- 📊 An interactive **Streamlit dashboard** for data science exploration
- 📓 A fully documented **Jupyter Notebook** with EDA & model training

Whether you're here for a college viva, portfolio review, or learning — this project has it all.

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| 🤖 **Multi-Model ML** | Trains & benchmarks XGBoost, Random Forest, SVM, Logistic Regression, Decision Tree |
| ⚖️ **SMOTE Balancing** | Handles class imbalance with Synthetic Minority Oversampling Technique |
| 💡 **Live BMI Calculator** | Client-side BMI computation as you type — no page reload needed |
| 🩺 **Health Recommendations** | AI-driven lifestyle & dietary suggestions per obesity class |
| 🎨 **Dual UI Architecture** | Both a Flask glassmorphism web app **and** a Streamlit data app |
| 🛡️ **Resilient Data Pipeline** | Generates synthetic training data automatically if Kaggle CSV is missing |
| 📊 **EDA Visualizations** | Auto-outputs heatmaps, distributions & feature importance charts |
| 📓 **Jupyter Notebook** | Complete EDA + ML workflow in a single reproducible notebook |

---

## 🏗️ Project Structure

```
Obesity_Prediction/
│
├── 📁 backend/
│   ├── app.py                   # Flask REST API (predictions endpoint)
│   ├── static/                  # CSS & JavaScript assets
│   └── templates/               # Glassmorphism HTML UI
│
├── 📁 frontend/
│   └── streamlit_app.py         # Interactive Streamlit dashboard
│
├── 📁 dataset/
│   └── ObesityDataSet.csv       # Training data (auto-generated if missing)
│
├── 📁 notebooks/
│   └── Obesity_Prediction_EDA_and_Modeling.ipynb
│
├── 📁 model/                    # Saved models (.pkl) — scaler, encoder, best_model
├── 📁 outputs/                  # EDA charts: heatmaps, confusion matrices
├── 📁 scripts/
│   └── generate_notebook.py     # Programmatically builds the Jupyter notebook
│
├── 🐍 train_pipeline.py         # Headless model training script
├── 📄 requirements.txt          # All Python dependencies
└── 📖 README.md
```

---

## 🧠 Machine Learning Pipeline

```
Raw Data ──► Feature Engineering ──► Preprocessing ──► SMOTE ──► Model Training
                                                                        │
                                                               ┌────────▼────────┐
                                                               │  Model Selection │
                                                               │  (Best F1/Acc)   │
                                                               └────────┬────────┘
                                                                        │
                                              Prediction ◄── best_model.pkl
```

### 🔍 Algorithms Compared

| Algorithm | Type |
|---|---|
| ⚡ XGBoost | Gradient Boosting |
| 🌲 Random Forest | Ensemble |
| 📐 Logistic Regression | Linear |
| 🔲 Support Vector Machine | Kernel-based |
| 🌿 Decision Tree | Tree-based |

### 🏷️ Obesity Classes Predicted

`Insufficient Weight` · `Normal Weight` · `Overweight Level I` · `Overweight Level II` · `Obesity Type I` · `Obesity Type II` · `Obesity Type III`

---

## 🚀 Quick Start

### Prerequisites
- Python **3.8+**
- `pip` package manager
- *(Optional)* Kaggle `ObesityDataSet.csv` in the `dataset/` folder

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/aditya484/Obesity-Prediction-System.git
cd Obesity-Prediction-System
```

### 2️⃣ Create & Activate Virtual Environment

**Windows**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the ML Models *(Do this first!)*

```bash
python train_pipeline.py
```

> 💡 This scans for `dataset/ObesityDataSet.csv`. If absent, it auto-generates synthetic data, trains all models, evaluates them, and saves `best_model.pkl` to `model/`.

---

## 🖥️ Running the Apps

### 🅰️ Flask Web App *(Recommended for Web-Dev / Vivas)*

```bash
python backend/app.py
```
Open in browser → **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

Features a stunning **glassmorphism UI** with live BMI calculation and health suggestions.

---

### 🅱️ Streamlit Dashboard *(Recommended for Data Science Demos)*

```bash
streamlit run frontend/streamlit_app.py
```

Opens automatically at **[http://localhost:8501](http://localhost:8501)**

---

### 📓 Jupyter Notebook *(EDA & Model Exploration)*

```bash
python generate_notebook.py
jupyter notebook notebooks/Obesity_Prediction_EDA_and_Modeling.ipynb
```

---

## 🌐 Deployment

### ☁️ Deploy Streamlit App *(Easiest — Free)*

1. Push this repo to GitHub
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Click **"New App"** → select your repo
4. Set main file path: `frontend/streamlit_app.py`
5. Click **Deploy** 🎉

### 🚂 Deploy Flask App on Render

1. Add `gunicorn` to `requirements.txt`
2. Push repo to GitHub
3. On **[Render.com](https://render.com)** → New Web Service → Connect repo
4. **Build Command:** `pip install -r requirements.txt`
5. **Start Command:** `gunicorn backend.app:app`

---

## 📦 Tech Stack

<div align="center">

| Layer | Technologies |
|---|---|
| **Language** | Python 3.8+ |
| **ML Libraries** | XGBoost, Scikit-learn, Imbalanced-learn (SMOTE) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Web Backend** | Flask |
| **Web Frontend** | Streamlit, HTML5, CSS3, JavaScript |
| **Model Storage** | Joblib (.pkl) |
| **Notebooks** | Jupyter, JupyterLab |

</div>

---

## 🤝 Connect with Me

<div align="center">

**Aditya Verma** — *Aspiring ML Engineer*

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aditya484/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/aditya484)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:adityalkverma484@gmail.com)

<br/>

*If you found this project helpful, please consider giving it a ⭐ — it motivates me to build more!*

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by [Aditya Verma](https://github.com/aditya484)

</div>
