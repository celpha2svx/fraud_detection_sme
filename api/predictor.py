import torch
import numpy as np
from DATA_PRE_FEATURE_ENG.model import FraudNet
import logging
from file_path import MODEL_PATH

logger = logging.getLogger(__name__)


class FraudPredictor:
    """Model inference wrapper."""

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Predictor initialized on {self.device}")

    def load_model(self, model_path=MODEL_PATH, input_dim=2196):
        """Load trained model."""
        try:
            self.model = FraudNet(input_dim=input_dim, hidden_dim=128, dropout=0.3)

            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Extract ONLY the model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def preprocess(self, transaction_data: dict) -> np.ndarray:
        """Convert transaction to model input format."""
        features = {
            'TransactionAmt': transaction_data['transaction_amount'],
            'TransactionAmt_log': np.log1p(transaction_data['transaction_amount']),
            'ProductCD': transaction_data['product_type'],
            'card_type': transaction_data.get('card_type', 'debit'),
            'email_domain': transaction_data.get('email_domain', 'unknown'),
            'hour': transaction_data['hour'],
            'day_of_week': transaction_data['day_of_week'],
            'is_business_hours': 1 if 9 <= transaction_data['hour'] <= 17 else 0,
            'is_weekend': 1 if transaction_data['day_of_week'] >= 5 else 0,
            'is_credit_card': 1 if transaction_data.get('card_type') == 'credit' else 0,
        }

        # Product risk scores (from your EDA)
        product_risk = {'C': 0.1169, 'S': 0.059, 'H': 0.0477, 'R': 0.0378, 'W': 0.0204}
        features['product_risk_score'] = product_risk.get(features['ProductCD'], 0.035)

        # Email risk
        high_risk_emails = ['mail.com', 'outlook.es', 'aim.com']
        features['is_high_risk_email'] = 1 if features['email_domain'] in high_risk_emails else 0

        # Create dummy feature vector (simplified for MVP)
        feature_vector = np.zeros(2196)
        feature_vector[0] = features['TransactionAmt_log']
        feature_vector[1] = features['product_risk_score']
        feature_vector[2] = features['is_business_hours']
        feature_vector[3] = features['is_weekend']
        feature_vector[4] = features['is_credit_card']

        return feature_vector.reshape(1, -1)

    def predict(self, transaction_data: dict) -> dict:
        """Generate fraud prediction."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            X = self.preprocess(transaction_data)
            X_tensor = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                prob = self.model(X_tensor).cpu().item()

            if prob < 0.3:
                risk_level = "LOW"
                message = "Transaction appears legitimate"
            elif prob < 0.6:
                risk_level = "MEDIUM"
                message = "Transaction flagged for review"
            else:
                risk_level = "HIGH"
                message = "High fraud probability - recommend blocking"

            return {
                'fraud_probability': round(prob, 4),
                'is_fraud': prob > 0.5,
                'risk_level': risk_level,
                'confidence': round(abs(prob - 0.5) * 2, 4),
                'message': message
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise