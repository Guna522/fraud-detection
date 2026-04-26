# Fraud Detection System (End-to-End ML Project)

A production-ready fraud detection system that predicts fraudulent transactions using machine learning and exposes predictions through a live API.

---

## Live Demo

👉 https://fraud-detection-production-6ca6.up.railway.app/docs

---

## Overview

This project builds a **real-time fraud detection system** using machine learning and deploys it as a public API.

Unlike basic ML projects, this includes:
- Model training
- Feature engineering
- API development
- Deployment

---

## Problem Statement

Credit card fraud detection is challenging because:
- Data is highly imbalanced (~0.17% fraud cases)
- Missing fraud (false negatives) is costly
- Real-time prediction is required

---

## Tech Stack

- **Machine Learning:** XGBoost, Scikit-learn  
- **Backend:** FastAPI  
- **Data Processing:** Pandas, NumPy  
- **Deployment:** Railway  
- **Containerization:** Docker  

---

## Model Details

- Algorithm: **XGBoost**
- Handled class imbalance using:
  - `scale_pos_weight`
- Focused on:
  - Recall (fraud detection rate)
  - PR-AUC (better for imbalanced data)

---

## Model Performance

| Metric | Score |
|------|------|
| ROC-AUC | ~0.97 |
| PR-AUC | ~0.87 |
| Recall (Fraud) | ~0.85 |

---

## 🔌 API Usage

### Endpoint
