from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    """Schema for incoming transaction data."""
    transaction_amount: float = Field(..., gt=0, description="Transaction amount")
    product_type: str = Field(..., description="Product code (W, C, H, R, S)")
    card_type: Optional[str] = Field("debit", description="Card type: credit/debit")
    email_domain: Optional[str] = Field(None, description="Customer email domain")
    hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Sunday)")

    class Config:
        model_config = {
            "json_schema_extra":{
            "example": {
                "transaction_amount": 15000.0,
                "product_type": "W",
                "card_type": "debit",
                "email_domain": "gmail.com",
                "hour": 14,
                "day_of_week": 2
            }
              }
        }


class FraudPrediction(BaseModel):
    """Schema for prediction response."""
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    is_fraud: bool = Field(..., description="Fraud flag (threshold=0.5)")
    risk_level: str = Field(..., description="LOW/MEDIUM/HIGH")
    confidence: float = Field(..., description="Model confidence")
    message: str = Field(..., description="Human-readable result")