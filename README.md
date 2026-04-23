# 🛡️ AI-Powered Spam Detection System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3-black)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-3-blue)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📌 Overview

An **end-to-end Machine Learning system** that classifies SMS messages as **SPAM** or **HAM (Normal)** with **98.3% accuracy**. This project demonstrates a complete ML pipeline from data preprocessing to deployment.

### 🎯 Key Achievements
- ✅ **98.3% accuracy** on 5,574 SMS messages
- ✅ **Real-time predictions** via Flask REST API
- ✅ **SQLite database** for storing prediction history
- ✅ **Interactive dashboard** with live statistics
- ✅ **Production-ready code** structure

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Spam Detection** | Classifies messages as SPAM or HAM instantly |
| 📊 **Live Statistics** | Shows total predictions, spam rate, confidence scores |
| 💾 **Database Storage** | All predictions saved with timestamps |
| 🎨 **Modern UI** | Responsive design with Bootstrap 5 |
| 📡 **REST API** | Easy integration with other applications |
| 📜 **History Tracking** | View recent predictions with details |

## 🛠️ Tech Stack

```mermaid
graph LR
    A[Frontend<br/>HTML/CSS/JS] --> B[Flask API]
    B --> C[ML Model<br/>Logistic Regression]
    B --> D[SQLite Database]
    C --> E[TF-IDF Vectorizer]
