# 🛡 Fraud Detection System for Nigerian SMEs

AI-powered real-time fraud detection for e-commerce transactions. Built with Deep Learning (PyTorch) and deployed as a REST API.

---

## 🎯 Features
- Real-time fraud scoring (<100ms response time)
- 88.5% AUC-ROC performance
- Simple REST API – integrate in minutes
- Web dashboard for manual checking
- Nigerian market optimized (Naira, debit cards, local patterns)

---

## 🚀 Quick Start

### 1. Install Dependencies
         pip install -r requirements.txt

### 2. Run API Server
     uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
### 3. Open Dashboard  
Open frontend/index.html in your browser

### 4. Test API
      curl -X POST "http://localhost:8000/predict" 
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