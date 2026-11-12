# 🏏 India Cricket Team Performance Prediction System Using Machine Learning

> **AI-Powered Indian Cricket Player Performance Prediction & Comparative Analytics System**  
> An end-to-end Machine Learning web application developed for an **Electronics & Communication Engineering (ECE) College DevOps Assignment**.

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1.3-black?logo=flask)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest%20Regressor-orange?logo=scikitlearn)
![Vercel](https://img.shields.io/badge/Deployment-Vercel%20Serverless-black?logo=vercel)
![Status](https://img.shields.io/badge/Build-Passing-brightgreen)

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Objectives](#-objectives)
3. [Key Features](#-key-features)
4. [Technologies Used](#-technologies-used)
5. [Project Folder Structure](#-project-folder-structure)
6. [Machine Learning Architecture](#-machine-learning-architecture)
7. [Installation & Setup](#-installation--setup)
8. [Local Execution](#-local-execution)
9. [Model Training & Retraining](#-model-training--retraining)
10. [GitHub Integration Guide](#-github-integration-guide)
11. [Vercel Deployment Guide](#-vercel-deployment-guide)
12. [Testing & Verification](#-testing--verification)
13. [Academic & Statistical Notice](#-academic--statistical-notice)

---

## 🌟 Project Overview
The **India Cricket Team Performance Prediction System** focuses specifically on the **Indian National Men's Cricket Team**. It provides data-driven performance impact scores (0–100) for Indian cricketers across all three formats of international cricket:
- 🔴 **Test Cricket** (Red Ball, 5-day format)
- ⚪ **One Day Internationals (ODI)** (White Ball, 50 Overs)
- ⚡ **T20 Internationals (T20I)** (White Ball, 20 Overs)

The system is powered by a **Scikit-Learn Random Forest Regressor** trained on verified international career data and calibrated against ICC player evaluation principles, sporting an **$R^2$ Score of 0.952**, **MAE of 2.42**, and **RMSE of 3.00**.

---

## 🎯 Objectives
- **Player Profiling:** Present detailed international career records for prominent Indian stars (Rohit Sharma, Virat Kohli, Jasprit Bumrah, Shubman Gill, Ravindra Jadeja, and more).
- **Format-Aware AI Forecasting:** Predict player performance scores (0–100) using 11 numerical and categorical features.
- **Head-to-Head Comparison:** Compare two Indian cricketers side-by-side using normalized radar and grouped bar charts.
- **Team Analytics:** Track aggregate runs, wickets, centuries, role distributions, and top performers.
- **DevOps Readiness:** Build a 100% serverless-compatible codebase ready for automated deployment via GitHub and Vercel.

---

## 🚀 Key Features

### 1. Interactive ML Prediction Form
- Supports both **one-click autofill** of verified Indian stars and **custom manual input**.
- Features 11 input attributes: Player Name, Role, Format, Matches, Runs, Batting Average, Strike Rate, Fours, Sixes, Wickets, Bowling Average, and Bowling Economy.
- Input validation and bounds checking.

### 2. Multi-Tier Impact Categorization
Predictions are mapped to four intuitive performance tiers:
- **80 – 100 (Excellent):** Elite Match Winner
- **60 – 79 (Good):** Reliable Core Performer
- **40 – 59 (Average):** Moderate Impact / Situational Performer
- **0 – 39 (Needs Improvement):** Underperforming / Developmental Phase

### 3. Client-Side Prediction History (localStorage)
- Prediction results are stored directly in browser `localStorage`.
- No local server disk writes, strictly adhering to serverless ephemeral constraints.
- Features individual record deletion and full history clearing.

### 4. Player Head-to-Head Comparison
- Side-by-side comparative matrix highlighting the statistical winner per metric.
- Dynamic **Chart.js Multi-Attribute Radar Chart** and **Grouped Bar Chart**.
- Formats: Test vs Test, ODI vs ODI, T20I vs T20I.

### 5. Indian Team Analytics Dashboard
- Leaderboards: Top Run Scorers & Top Wicket Takers.
- Squad composition doughnut chart.
- Multi-format specialization cards.

---

## 💻 Technologies Used

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Backend** | Python 3.10+, Flask 3.1 | REST API, Template Routing, WSGI Server |
| **Machine Learning** | Scikit-Learn (Random Forest) | Non-linear regression score modeling |
| **Data Processing** | Pandas, NumPy | Statistical modeling, array transformations |
| **Model Persistence** | Joblib | Model serialization & fast inference loading |
| **Frontend** | HTML5, CSS3, JavaScript (ES6) | Responsive, mobile-first cricket dashboard |
| **Visualizations** | Chart.js 4.4 | Interactive Radar, Horizontal Bar, & Doughnut charts |
| **DevOps / CI/CD** | Git, GitHub, Vercel | Version control & serverless cloud deployment |

---

## 📁 Project Folder Structure

```text
India-Cricket-Performance-Prediction/
│
├── api/
│   └── index.py               # Flask application & Vercel serverless entry point
│
├── model/
│   ├── train_model.py         # ML training, evaluation & artifact exporter script
│   ├── cricket_rf_model.joblib# Serialized Random Forest model artifact
│   └── model_metrics.json     # Quantitative evaluation metrics & feature weights
│
├── data/
│   ├── README.md              # Dataset documentation & data source notice
│   ├── players.json           # Verified Indian national squad career statistics
│   └── cricket_training_dataset.csv # Compiled training dataset (real + synthetic benchmarks)
│
├── templates/
│   ├── base.html              # Base HTML layout with responsive navbar & footer
│   ├── index.html             # Homepage with cricket stadium hero & features
│   ├── players.html           # Indian squad directory with search & format tabs
│   ├── prediction.html        # Interactive ML prediction form, gauge & history
│   ├── comparison.html        # Player comparison with radar & bar charts
│   ├── dashboard.html         # Team analytics dashboard & leaderboards
│   └── about.html             # Project documentation, ML evaluation & DevOps pipeline
│
├── static/
│   ├── css/
│   │   └── style.css          # Indian cricket theme (navy, blue, saffron, white cards)
│   ├── js/
│   │   └── script.js          # Prediction engine, Chart.js, and localStorage handler
│   └── images/
│       ├── cricket-crest.svg  # Legal custom cricket crest logo
│       └── stadium-hero.svg   # Custom cricket stadium backdrop
│
├── app.py                     # Root runner script for convenient local execution
├── requirements.txt           # Python package dependencies
├── vercel.json                # Vercel serverless deployment routing config
├── .gitignore                 # Git ignore rules for Python, IDEs & logs
└── README.md                  # Comprehensive assignment documentation
```

---

## 🧠 Machine Learning Architecture

### Algorithm: Random Forest Regressor
- **Estimators (`n_estimators`):** 150 decision trees
- **Maximum Depth (`max_depth`):** 12
- **Split Quality:** Mean Squared Error (MSE)
- **Train/Test Split:** 80% Training (1,961 samples), 20% Testing (491 samples)

### Quantitative Evaluation Metrics

| Metric | Overall Score | Test Format | ODI Format | T20I Format |
| :--- | :---: | :---: | :---: | :---: |
| **$R^2$ Score** | **0.9521** | 0.946 | 0.950 | 0.946 |
| **Mean Absolute Error (MAE)** | **2.4265** | 2.463 | 2.293 | 2.524 |
| **Root Mean Squared Error (RMSE)**| **2.9998** | 2.973 | 2.888 | 3.152 |

### Top 5 Feature Importances
1. **Batting Average (`batting_avg`):** 45.48%
2. **Strike Rate (`strike_rate`):** 31.87%
3. **Player Role (`role_encoded`):** 13.03%
4. **Wickets Taken (`wickets`):** 4.09%
5. **Bowling Average (`bowling_avg`):** 2.60%

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed.
- Git installed on your system.

### Step 1: Clone or Navigate to the Workspace
```bash
cd "c:\Users\lokes\OneDrive\Desktop\INDIAN CRICKET TEAM PERFORMANCE PREDICTION"
```

### Step 2: (Optional) Create and Activate a Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Local Execution

You can start the web server using either entry point:

### Method 1: Using the root runner
```bash
python app.py
```

### Method 2: Using the API entry point
```bash
python api/index.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🔄 Model Training & Retraining

To train the Random Forest model from scratch and recalculate all evaluation metrics:

```bash
python model/train_model.py
```

This script:
1. Loads verified squad records from `data/players.json`.
2. Augments the dataset with labelled synthetic benchmarks to ensure robust continuous generalization across all format & role combinations.
3. Saves the dataset to `data/cricket_training_dataset.csv`.
4. Trains the `RandomForestRegressor`.
5. Evaluates overall and format-specific MAE, RMSE, and $R^2$ scores.
6. Serializes the updated model to `model/cricket_rf_model.joblib`.
7. Exports updated metrics to `model/model_metrics.json`.

---

## 🐙 GitHub Integration Guide

Follow these steps to push the project to your GitHub repository:

### Step 1: Initialize Git Repository
```bash
git init
```

### Step 2: Add Files & Make Initial Commit
```bash
git add .
git commit -m "feat: initial commit of India Cricket Performance Prediction System"
```

### Step 3: Link to Your GitHub Repository
Create a new public repository on GitHub (e.g. `India-Cricket-Performance-Prediction`), then link and push:

```bash
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/India-Cricket-Performance-Prediction.git
git push -u origin main
```

---

## 🚀 Vercel Deployment Guide

The project is structured to deploy smoothly to Vercel as a Python serverless function:

### Step 1: Install Vercel CLI (or use Vercel Web Dashboard)
```bash
npm install -g vercel
```

### Step 2: Deploy from Terminal
```bash
vercel
```
- Select your Vercel account.
- Link to existing project? **No**
- What's your project name? **india-cricket-performance-prediction**
- In which directory is your code located? **./**

### Step 3: Production Deployment
```bash
vercel --prod
```

### Alternative: Deploy via GitHub (Recommended)
1. Go to [vercel.com](https://vercel.com) and log in.
2. Click **"Add New Project"** → **"Import Git Repository"**.
3. Select your GitHub repository (`India-Cricket-Performance-Prediction`).
4. Keep the default settings (`vercel.json` handles routing automatically).
5. Click **"Deploy"**.
6. Vercel will build and assign you a live production URL (e.g., `https://india-cricket-performance-prediction.vercel.app`).

---

## 🧪 Testing & Verification

Before submitting the DevOps assignment, perform these verification checks:

1. **Route Testing:**
   - `http://127.0.0.1:5000/` (Home page loads with hero, format cards, and features)
   - `http://127.0.0.1:5000/players` (Profiles load, search filter works)
   - `http://127.0.0.1:5000/prediction` (Prediction form autofills on player select)
   - `http://127.0.0.1:5000/comparison` (Comparison radar and bar charts render)
   - `http://127.0.0.1:5000/dashboard` (Leaderboards and team charts render)
   - `http://127.0.0.1:5000/about` (Evaluation metrics and DevOps documentation render)

2. **Prediction Verification:**
   - Select **Virat Kohli** with **ODI** format. Click **Run Machine Learning Prediction**.
   - Verify that the score calculates (~91.8/100, Tier: **Excellent**).
   - Check that history updates in the table below.
   - Refresh the page and confirm the prediction persists in `localStorage`.

3. **Comparison Verification:**
   - Select **Rohit Sharma** and **Virat Kohli** in **ODI**.
   - Click **Compare Players** and verify winner highlights and radar charts.

---

## 🛡️ Academic & Statistical Notice
- **Data Source:** Verified international career records compiled from official ICC and ESPNcricinfo sources.
- **Academic Fallback Notice:** Synthetic data points generated during training serve strictly as statistical benchmarks to establish smooth regression boundaries across all format/role combinations.
- **Disclaimer:** Predicted scores represent statistical impact ratings based on historical metrics and do not guarantee future match outcomes.

---

**Developed for ECE College DevOps Assignment • 2026**

