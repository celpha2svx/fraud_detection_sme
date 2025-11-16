import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix,
                             precision_score, recall_score, f1_score, matthews_corrcoef)
import logging
import gc

# --- Setup Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Helper Functions ---

def clean_sparse_matrix(X_sparse):
    """
    Clean sparse matrix in-place without converting to dense.
    This is memory-efficient for large sparse matrices.
    """
    if not sp.issparse(X_sparse):
        X_sparse = sp.csr_matrix(X_sparse)

    # Only clean the actual stored values (non-zero elements)
    X_sparse.data = np.nan_to_num(X_sparse.data, nan=0.0, posinf=0.0, neginf=0.0)
    X_sparse.eliminate_zeros()  # Remove any zeros created

    logger.info(f"Sparse matrix cleaned | Shape: {X_sparse.shape} | "
                f"Non-zero: {X_sparse.nnz:,} | "
                f"Sparsity: {(1 - X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1])):.2%}")

    return X_sparse


# --- Custom Dataset ---
class SparseTensorDataset(Dataset):
    def __init__(self, X_sparse, y_tensor):
        """
        Dataset that handles sparse matrices efficiently.
        Converts rows to dense only when needed (during batch loading).
        """
        self.X_sparse = X_sparse.tocsr() if sp.issparse(X_sparse) else sp.csr_matrix(X_sparse)
        self.y_tensor = y_tensor

    def __len__(self):
        return self.X_sparse.shape[0]

    def __getitem__(self, idx):
        # Only convert single row to dense
        x_dense_np = self.X_sparse[idx].toarray().astype(np.float32)
        x_dense = torch.from_numpy(x_dense_np).squeeze(0)
        y = self.y_tensor[idx].item()
        return x_dense, y


# --- Neural Network Model ---
class FraudNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        # Conservative initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=0.5)
        nn.init.xavier_normal_(self.fc2.weight, gain=0.5)
        nn.init.xavier_normal_(self.fc3.weight, gain=0.5)

        # Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)  # logits
        return x


# --- Model Controller ---
class FraudModel:
    def __init__(self, input_dim, scaler=None, hidden_dim=128, dropout=0.3,
               default_lr=1e-4, chunk_size=10000):
        self.input_dim = input_dim
        # Use MaxAbsScaler - works directly on sparse matrices!
        self.scaler = scaler if scaler is not None else MaxAbsScaler()
        self.chunk_size = chunk_size
        self.model = FraudNet(input_dim, hidden_dim, dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=default_lr, weight_decay=1e-5)

        logger.info(f"Model initialized on {self.device}")
        logger.info(f"Using MaxAbsScaler (sparse-friendly)")
        logger.info(f"Architecture: {input_dim} -> {hidden_dim} -> {hidden_dim // 2} -> 1")
        logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def set_learning_rate(self, lr):
        """Change learning rate dynamically"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        logger.info(f"Learning rate set to {lr}")

    def _prepare_data(self, X, y=None, fit_scaler=False):
        """
        Memory-efficient data preparation using MaxAbsScaler.
        MaxAbsScaler works DIRECTLY on sparse matrices - no dense conversion!
        """
        logger.info(f"Preparing data | fit_scaler={fit_scaler}")

        # Step 1: Ensure it's sparse and clean it
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X_clean = clean_sparse_matrix(X)

        # Step 2: Fit or transform with MaxAbsScaler (NO CHUNKING NEEDED!)
        if fit_scaler:
            logger.info("Fitting MaxAbsScaler...")
            X_scaled = self.scaler.fit_transform(X_clean)
            logger.info(f"MaxAbsScaler fitted | Max absolute value per feature computed")
        else:
            logger.info("Transforming with fitted MaxAbsScaler...")
            X_scaled = self.scaler.transform(X_clean)

        # Step 3: Clip extreme values if any (still sparse)
        # MaxAbsScaler already scales to [-1, 1], so we just ensure no overflow
        X_scaled.data = np.clip(X_scaled.data, -10, 10)

        logger.info(f"Data prepared | Shape: {X_scaled.shape}")
        logger.info(f"Sparsity: {(1 - X_scaled.nnz / (X_scaled.shape[0] * X_scaled.shape[1])):.2%}")

        # Clear memory
        del X_clean
        gc.collect()

        return X_scaled, y

    def train(self, X_train, y_train, X_val, y_val,
              epochs=5, batch_size=1024, clip_norm=1.0, lr=None):
        """
        Train the fraud detection model with memory-efficient processing.
        """
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)

        if lr is not None:
            self.set_learning_rate(lr)

        # Prepare training data (fit scaler here)
        logger.info("Preparing training data...")
        X_train_processed, _ = self._prepare_data(X_train, y_train, fit_scaler=True)

        # Prepare validation data (use fitted scaler)
        logger.info("Preparing validation data...")
        X_val_processed, _ = self._prepare_data(X_val, y_val, fit_scaler=False)

        # Create PyTorch dataset and loader
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        train_dataset = SparseTensorDataset(X_train_processed, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)  # num_workers=0 for Windows

        logger.info(f"Training batches: {len(train_loader)}")

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            grad_norms = []
            batch_count = 0

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.float().unsqueeze(1).to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)

                # Log first batch of each epoch for monitoring
                if batch_idx == 0:
                    logger.info(f"Epoch {epoch + 1} Batch 0 outputs | "
                                f"min: {outputs.min().item():.4f}, "
                                f"max: {outputs.max().item():.4f}, "
                                f"mean: {outputs.mean().item():.4f}")

                # Safety check for NaN/Inf
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    logger.error("NaN or Inf detected in model output!")
                    logger.error("Diagnosing issue...")
                    for name, param in self.model.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            logger.error(f"Issue found in parameter: {name}")
                    return self

                # Compute loss
                loss = self.criterion(outputs, batch_y)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"NaN or Inf loss at batch {batch_idx}! Stopping training.")
                    return self

                # Backward pass
                loss.backward()

                # Calculate gradient norm before clipping
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                grad_norms.append(total_norm)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)

                # Update weights
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

                # Log every 50 batches
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch + 1} Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")

            # Epoch summary
            avg_loss = epoch_loss / batch_count
            avg_grad_norm = np.mean(grad_norms)
            val_auc = self._validate(X_val_processed, y_val)

            logger.info(f"Epoch {epoch + 1}/{epochs} | "
                        f"Loss: {avg_loss:.6f} | "
                        f"Val AUC: {val_auc:.4f} | "
                        f"Grad Norm: {avg_grad_norm:.4f}")

            # Clear memory
            gc.collect()

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        return self

    def _validate(self, X_val_processed, y_val, batch_size=5000):
        """
        Internal validation during training.
        Process in batches to avoid memory issues.
        """
        self.model.eval()
        all_probs = []

        n_samples = X_val_processed.shape[0]

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)

                # Get batch as dense
                batch_X = X_val_processed[start_idx:end_idx].toarray().astype(np.float32)
                batch_X_tensor = torch.from_numpy(batch_X).to(self.device)

                # Get predictions
                batch_logits = self.model(batch_X_tensor).cpu().numpy().flatten()
                batch_probs = 1 / (1 + np.exp(-np.clip(batch_logits, -20, 20)))

                all_probs.extend(batch_probs)

                # Clear memory
                del batch_X, batch_X_tensor, batch_logits

        all_probs = np.array(all_probs)
        return roc_auc_score(y_val, all_probs)

    def evaluate(self, X_val, y_val, threshold=0.5):
        """
        Comprehensive evaluation with metrics.
        Processes in batches to handle large validation sets.
        """
        logger.info("=" * 60)
        logger.info("STARTING EVALUATION")
        logger.info("=" * 60)

        self.model.eval()

        # Prepare validation data
        X_val_processed, _ = self._prepare_data(X_val, y_val, fit_scaler=False)

        # Get predictions in batches
        all_probs = []
        n_samples = X_val_processed.shape[0]
        batch_size = 5000

        logger.info(f"Generating predictions for {n_samples:,} samples...")

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)

                batch_X = X_val_processed[start_idx:end_idx].toarray().astype(np.float32)
                batch_X_tensor = torch.from_numpy(batch_X).to(self.device)

                batch_logits = self.model(batch_X_tensor).cpu().numpy().flatten()
                batch_probs = 1 / (1 + np.exp(-np.clip(batch_logits, -20, 20)))

                all_probs.extend(batch_probs)

                del batch_X, batch_X_tensor, batch_logits

                if start_idx % (batch_size * 10) == 0:
                    logger.info(f"Processed {end_idx}/{n_samples} samples")

        val_probs = np.array(all_probs)
        val_preds = (val_probs > threshold).astype(int)

        # Display results
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(classification_report(y_val, val_preds, target_names=['Legit', 'Fraud']))

        auc = roc_auc_score(y_val, val_probs)
        precision = precision_score(y_val, val_preds)
        recall = recall_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds)
        mcc = matthews_corrcoef(y_val, val_preds)

        print(f"\nKey Metrics:")
        print(f"  AUC-ROC:   {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  MCC:       {mcc:.4f}")

        cm = confusion_matrix(y_val, val_preds)
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0, 0]:>6} | FP: {cm[0, 1]:>6}")
        print(f"  FN: {cm[1, 0]:>6} | TP: {cm[1, 1]:>6}")
        print("=" * 60 + "\n")

        return {
            'auc': auc,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc
        }

    def predict(self, X, threshold=0.5):
        """
        Make predictions on new data in batches.
        Returns probabilities by default.
        """
        self.model.eval()

        # Prepare data
        X_processed, _ = self._prepare_data(X, fit_scaler=False)

        # Get predictions in batches
        all_probs = []
        n_samples = X_processed.shape[0]
        batch_size = 5000

        logger.info(f"Generating predictions for {n_samples:,} samples...")

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)

                batch_X = X_processed[start_idx:end_idx].toarray().astype(np.float32)
                batch_X_tensor = torch.from_numpy(batch_X).to(self.device)

                batch_logits = self.model(batch_X_tensor).cpu().numpy().flatten()
                batch_probs = 1 / (1 + np.exp(-np.clip(batch_logits, -20, 20)))

                all_probs.extend(batch_probs)

                del batch_X, batch_X_tensor, batch_logits

        return np.array(all_probs)

    def save(self, path='fraud_model.pth'):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path='fraud_model.pth'):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler = checkpoint['scaler']
        logger.info(f"Model loaded from {path}")

