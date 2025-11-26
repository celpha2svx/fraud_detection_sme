# 🛡️ FRAUD_DETECTION_SYSTEM

*AI-Powered Real-Time Fraud Prevention for Nigerian E-Commerce*

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)

---

## 📘 Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [API Testing](#api-testing)
- [Features](#features)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Use Cases](#use-cases)
- [Roadmap](#roadmap)
- [License](#license)

---

## 🧠 Overview

Fraud Detection System is a sophisticated deep learning solution specifically designed for Nigerian SMEs and e-commerce platforms. Built with PyTorch and deployed as a high-performance REST API, it provides real-time fraud detection with sub-100ms response times, helping businesses protect their revenue and customers.

## Why Fraud Detection System?

Nigerian e-commerce faces unique fraud challenges, from card-not-present transactions to local payment patterns. This system addresses these specific needs through:

- **Lightning-Fast Detection**: Real-time fraud scoring with <100ms response time ensures seamless checkout experiences.

- **Nigerian Market Optimization**: Trained on local transaction patterns, supporting Naira currency and Nigerian debit card behaviors.

- **Production-Ready Architecture**: Built with FastAPI for scalability, featuring both automated API integration and manual review dashboards.

- **High Accuracy**: Achieves 88.5% AUC-ROC performance on 590K transactions, balancing fraud detection with legitimate transaction approval.

- **Developer-Friendly Integration**: Simple REST API enables integration in minutes, with comprehensive documentation and examples.

- **Comprehensive Analytics**: Interactive web dashboard provides transaction insights, fraud patterns, and manual review capabilities.

---

## 🚀 Getting Started

### ✅ Prerequisites

This project requires the following dependencies:

- **Programming Language**: Python 3.8+
- **Package Manager**: Pip

---

### 🛠 Installation

Build the Fraud Detection System from source and install dependencies:

```bash
# Clone the repository
git clone https://github.com/celpha2svx/fraud_detection_system

# Navigate to the project directory
cd fraud_detection_system

# Install the dependencies
pip install -r requirements.txt
```

---

### ▶ Usage

Run the API server with:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Access the web dashboard by opening `frontend/index.html` in your browser.

---

### 🧪 API Testing

Test the fraud detection endpoint with a sample transaction:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 15000,
    "product_type": "W",
    "card_type": "debit",
    "hour": 14,
    "day_of_week": 2
  }'
```

---

## ✨ Features

- **Real-Time Fraud Scoring**: Sub-100ms response time for seamless transaction processing
- **High-Performance ML Model**: 88.5% AUC-ROC accuracy on Nigerian transaction data
- **REST API Integration**: Simple, developer-friendly API for quick integration
- **Interactive Web Dashboard**: Manual transaction review and fraud pattern visualization
- **Nigerian Market Specialization**: Optimized for Naira currency, local card types, and regional patterns
- **Scalable Architecture**: Built with FastAPI and PyTorch for production deployment

---

## 📊 Model Performance

The system demonstrates strong predictive capabilities:

- **AUC-ROC**: 0.8855
- **Precision**: 58.4%
- **Recall**: 33.5% (v1.0 – improvements targeted for v2.0)
- **Training Dataset**: 590,000+ real-world transactions
- **Response Time**: <100ms average

---

## 📁 Project Structure

```
fraud_detection/
├── api/                    # FastAPI backend and endpoints
├── frontend/               # Web dashboard (HTML, CSS, JS)
├── EDA_Detection/          # Exploratory data analysis notebooks
├── DATA_PRE_FEATURE_ENG/   # Data preprocessing and neural network
├── docs/                   # Integration guides and documentation
├── Files/                  # Training data (train_id, train_trans)
├── file_path.py            # File path management pipeline
├── requirements.txt        # Python dependencies
```

---

## 🔧 Technology Stack

- **Machine Learning**: PyTorch, Scikit-learn, SMOTE
- **API Framework**: FastAPI, Uvicorn
- **Data Processing**: Pandas, NumPy, SciPy
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Model Training**: Deep Learning with PyTorch

---

## 🎯 Use Cases

- **E-Commerce Checkout Validation**: Real-time fraud checks during payment processing
- **POS Transaction Monitoring**: Point-of-sale fraud detection for retail businesses
- **Payment Gateway Integration**: Seamless integration with Nigerian payment processors
- **Manual Transaction Review**: Dashboard-based investigation of suspicious transactions
- **Risk Assessment**: Transaction scoring for risk management teams

---

## 📈 Roadmap

Future enhancements planned for the system:

- **Improved Recall**: Enhanced model training to catch more fraudulent transactions
- **API Authentication**: Secure API keys and authentication mechanisms
- **Cloud Deployment**: Production deployment on Railway or Render
- **Explainability Features**: SHAP values and fraud reason reporting
- **Nigerian Dataset Expansion**: Continuous retraining with local fraud patterns
- **Multi-Currency Support**: Expansion to other African currencies

---

## 📜 License

This project is licensed under the MIT License – Free for commercial use

---

## 👨‍💻 Author

**Developed by Ademuyiwa Afeez** ✨

*Data Scientist | Building Fraud Prevention for African SMEs*

For contributions, issues, or feature requests, please open a GitHub issue or pull request.

---      curl -X POST "http://localhost:8000/predict" 
     -H "Content-Type: application/json" 
     -d '{"transaction_amount": 15000, "product_type": "W", "card_type": "debit", "hour": 14, "day_of_week": 2}'

 ---

## 📊 Model Performance
- AUC-ROC: 0.8855  
- Precision: 58.4%  
- Recall: 33.5% (v1.0 – improving in v2.0)  
- Training Data: 590K transactions

---

## 📁 Project Structure
    fraud_detection/
    ├── api/                    # FastAPI backend
    ├── frontend/               # Web dashboard
    ├── EDA_Detection/          # Exploratory data analysis
    ├── DATA_PRE_FEATURE_ENG/   # Data preprocessing|Neural network
    ├── docs/                   # Integration guides
    ├── Files/                  # CSV files(train_id,train_trans)
    ├── file_path.py            # pipeline for files path
    ├── requirements.txt        # required packages

---

## 🔧 Technology Stack
- ML: PyTorch, Scikit-learn, SMOTE  
- API: FastAPI, Uvicorn  
- Data: Pandas, NumPy, SciPy  
- Frontend: Vanilla JS, HTML5, CSS3  

---

## 🎯 Use Cases
- E-commerce checkout validation  
- POS transaction monitoring  
- Payment gateway integration  
- Manual transaction review  

---

## 📈 Roadmap
- Improve recall (catch more frauds)  
- Add authentication (API keys)  
- Deploy to cloud (Railway/Render)  
- Add explainability (why flagged)  
- Collect Nigerian fraud data for retraining  

---

## 👨‍💻 Author
Ademuyiwa Afeez 

Data Scientist | Building fraud prevention for African SMEs

---

## 📝 License
MIT License – Free for commercial use
