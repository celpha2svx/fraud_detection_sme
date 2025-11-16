import pandas as pd
import numpy as np
import gc
from EDA_Detection.cleaning_func import reduce_mem_usage
from file_path import TRAIN_ID,TRAIN_TRANS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack, csr_matrix

# ===== STEP 2.1: MISSING VALUES STRATEGY =====
class FraudDetectionPipeline:
    """
    A pipeline class for loading, merging, and preprocessing the Fraud Detection data.
    """

    def __init__(self, trans_path, id_path, high_missing_threshold=80):
        self.trans_path = trans_path
        self.id_path = id_path
        self.missing_threshold = high_missing_threshold
        self.df = None
        print("Pipeline initialized.")

    def _get_loading_params(self):
        """Defines columns and dtypes for essential data loading."""
        trans_cols_to_keep = [
            'TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt',
            'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'dist1', 'dist2'
        ]
        # Add C, D, and M columns
        trans_cols_to_keep.extend([f'C{i}' for i in range(1, 15)])
        trans_cols_to_keep.extend([f'D{i}' for i in range(1, 16)])
        trans_cols_to_keep.extend([f'M{i}' for i in range(1, 10)])

        # Define dtypes for compatibility (using float32)
        dtypes_trans_essentials = {
            'TransactionAmt': 'float32', 'isFraud': 'int8', 'TransactionDT': 'int32',
            'TransactionID': 'int32', 'ProductCD': 'category', 'card1': 'int16',
            'card2': 'float32', 'card3': 'float32', 'card4': 'category',
            'card5': 'float32', 'card6': 'category', 'addr1': 'float32',
            'addr2': 'float32', 'dist1': 'float32', 'dist2': 'float32',
            'P_emaildomain': 'category', 'R_emaildomain': 'category'
        }
        dtypes_trans_essentials.update({col: 'float32' for col in [f'C{i}' for i in range(1, 15)]})
        dtypes_trans_essentials.update({col: 'float32' for col in [f'D{i}' for i in range(1, 16)]})
        dtypes_trans_essentials.update({col: 'category' for col in [f'M{i}' for i in range(1, 10)]})

        return trans_cols_to_keep, dtypes_trans_essentials

    def load_and_merge(self):
        """Loads essential data and merges Transaction and Identity files."""
        print("\n" + "=" * 50)
        print("STEP 1: Data Loading & Merging")
        print("=" * 50)

        trans_cols_to_keep, dtypes_trans_essentials = self._get_loading_params()

        print("-> Loading train_transaction.csv (Essential Columns)...")
        train_trans = pd.read_csv(self.trans_path, dtype=dtypes_trans_essentials, usecols=trans_cols_to_keep)
        train_trans = reduce_mem_usage(train_trans)  # Optimize memory AFTER loading

        print("\n-> Loading train_identity.csv...")
        train_id = pd.read_csv(self.id_path, dtype={'TransactionID': 'int32'})
        train_id = reduce_mem_usage(train_id)

        print("\n-> Merging dataframes...")
        self.df = train_trans.merge(train_id, on='TransactionID', how="left")

        # Clean up memory immediately
        del train_trans, train_id
        gc.collect()

        print(f"Merge successful. Final shape: {self.df.shape}")
        return self

    def handle_missing_values(self):
        """Drops high-missing columns, creates indicators, and fills NaNs."""
        if self.df is None:
            raise ValueError("Data not loaded. Please run load_and_merge() first.")

        print("\n" + "=" * 50)
        print("STEP 2: Handle Missing Values")
        print("=" * 50)

        # 1. Identify and Drop columns with excessive missing values
        missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
        high_missing = missing_pct[missing_pct > self.missing_threshold]
        cols_to_drop = [col for col in high_missing.index.tolist() if col not in ['TransactionID', 'isFraud']]

        print(f"-> Dropping {len(cols_to_drop)} columns with >{self.missing_threshold}% missing data...")
        self.df = self.df.drop(columns=cols_to_drop)
        print(f"\tShape after dropping: {self.df.shape}")

        # 2. For remaining missing values, create indicator features
        important_cols_with_missing = ['P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo']
        for col in important_cols_with_missing:
            if col in self.df.columns and self.df[col].isnull().any():
                self.df[f'{col}_missing'] = self.df[col].isnull().astype(np.int8)
                print(f"\tCreated indicator: {col}_missing")

        # 3. Fill missing values appropriately

        # --- CATEGORICAL FILL FIX ---
        cat_cols = self.df.select_dtypes(include=['category']).columns
        for col in cat_cols:
            if self.df[col].isnull().any():
                # FIX: Add 'unknown' to categories BEFORE filling
                if 'unknown' not in self.df[col].cat.categories:
                    self.df[col] = self.df[col].cat.add_categories('unknown')

                self.df[col] = self.df[col].fillna('unknown')
                print(f"\tFilled categorical {col} with 'unknown'")

        # Numerical columns: fill with median
        num_cols = self.df.select_dtypes(include=[np.float16, np.float32, np.int16, np.int32]).columns
        for col in num_cols:
            if col != 'isFraud' and self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                print(f"\tFilled numerical {col} with median: {median_val}")

        final_missing_count = self.df.isnull().sum().sum()
        print(f"\nFinal missing values check (Should be 0): {final_missing_count}")

        gc.collect()
        return self

    def create_time_features(self):
        """Example of a feature engineering step."""
        if self.df is None:
            raise ValueError("Data not loaded. Please run load_and_merge() first.")

        print("\n" + "=" * 50)
        print("STEP 2.1: Feature Engineering (Time Features)")
        print("=" * 50)

        # TransactionDT is seconds from an arbitrary point
        # Create day of week and hour of day features
        START_HOUR = 4  # Common assumption for the start of the 'day' in this dataset

        self.df['DOW'] = np.floor((self.df['TransactionDT'] / (3600 * 24)) % 7)
        self.df['HOUR'] = np.floor((self.df['TransactionDT'] / 3600) % 24)

        print("\tCreated DOW (Day of Week) and HOUR features.")
        return self

    def engineer_features(self):
        """Creates Nigerian-aware, relative features."""
        if self.df is None:
            raise ValueError("Data not loaded.")

        print("\n" + "=" * 50)
        print("STEP 2.2: Advanced Feature Engineering")
        print("=" * 50)

        # 1. AMOUNT FEATURES (Relative, not absolute)
        print("-> Creating amount features...")
        self.df['TransactionAmt_log'] = np.log1p(self.df['TransactionAmt'])

        # Amount deviation by product type
        self.df['amt_product_mean'] = self.df.groupby('ProductCD')['TransactionAmt'].transform('mean')
        self.df['amt_deviation_from_product'] = self.df['TransactionAmt'] / (self.df['amt_product_mean'] + 1)

        # 2. TIME FEATURES (Behavioral, not hardcoded)
        print("-> Creating time features...")
        # Business hours flag (9am-5pm concept, model learns specifics)
        self.df['is_business_hours'] = ((self.df['HOUR'] >= 9) & (self.df['HOUR'] <= 17)).astype(np.int8)

        # Weekend flag
        self.df['is_weekend'] = (self.df['DOW'] >= 5).astype(np.int8)

        # 3. PRODUCT RISK SCORING
        print("-> Creating product risk scores...")
        product_fraud_rate = self.df.groupby('ProductCD')['isFraud'].mean()
        self.df['product_risk_score'] = self.df['ProductCD'].map(product_fraud_rate)

        # 4. EMAIL DOMAIN RISK
        print("-> Creating email risk features...")
        # High-risk domains from EDA
        high_risk_domains = ['mail.com', 'outlook.es', 'aim.com']
        self.df['is_high_risk_email'] = self.df['P_emaildomain'].isin(high_risk_domains).astype(np.int8)

        # Email domain fraud rate
        email_fraud_rate = self.df.groupby('P_emaildomain')['isFraud'].mean()
        self.df['email_risk_score'] = self.df['P_emaildomain'].map(email_fraud_rate)

        # 5. CARD TYPE RISK (Conceptual for Nigerian adaptation)
        print("-> Creating card risk features...")
        # Credit=1, Debit=0 (we'll adapt this for Nigerian payment types later)
        self.df['is_credit_card'] = (self.df['card6'] == 'credit').astype(np.int8)

        # 6. INTERACTION FEATURES (High-risk combinations)
        print("-> Creating interaction features...")
        # Product C + Credit Card = extra risky
        self.df['risky_combo'] = ((self.df['ProductCD'] == 'C') &
                                  (self.df['card6'] == 'credit')).astype(np.int8)

        # 7. DROP TEMPORARY COLUMNS
        self.df = self.df.drop(columns=['amt_product_mean'])

        print(f"\nFeature engineering complete! New shape: {self.df.shape}")
        gc.collect()
        return self

    def prepare_for_training(self, test_size=0.2, random_state=42):
        """Splits data, applies SMOTE, scales features using memory-safe sparse methods."""
        if self.df is None:
            raise ValueError("Data not loaded.")

        print("\n" + "=" * 50)
        print("STEP 2.3: Train/Val Split + SMOTE + Scaling (FINAL MEMORY-SAFE FIX)")
        print("=" * 50)

        # 1. Separate features and target
        print("-> Separating features and target...")
        df = self.df.drop(columns=['TransactionID']).copy()
        y = df['isFraud']
        X = df.drop(columns=['isFraud'])
        del df  # Free up memory
        gc.collect()

        # 2. Identify Column Types
        cat_cols = X.select_dtypes(include=['category']).columns.tolist()
        num_cols = X.select_dtypes(include=[np.float16, np.float32, np.int16, np.int32, np.int8]).columns.tolist()

        print(f"-> Found {len(num_cols)} numeric and {len(cat_cols)} categorical columns.")

        # 3. Memory-Safe Categorical Encoding & Sparse Conversion
        print("-> Encoding categorical features using memory-safe sparse conversion...")

        sparse_matrix_list = []

        # Process Numeric Data: Convert directly to CSR (should be safe as this is smaller)
        X_numeric = X[num_cols].copy()
        X_numeric = np.nan_to_num(X_numeric,
                                  nan=0,
                                  posinf=np.finfo(np.float32).max,
                                  neginf=np.finfo(np.float32).min )

        # We must ensure X_numeric is free of NaNs before conversion (it should be after step 2.1)
        # If NaNs exist, csr_matrix conversion can be problematic. Assume clean data from step 2.1.
        X_numeric_sparse = csr_matrix(X_numeric.astype(np.float32))  # Use float32 for safety
        sparse_matrix_list.append(X_numeric_sparse)

        # Process Categorical Data: Convert categories to codes, then use Feature Mapping concept
        for col in cat_cols:
            # Convert category labels to numerical codes (+1 to reserve 0 for missing/default)
            # This converts a single column into a single column of integers
            codes = X[col].cat.codes.values + 1

            # Create a basic sparse representation (mimicking one-hot, but memory safe)
            # Row indices are 0 to N, Column indices are the category codes
            n_rows = len(X)
            n_cols = len(X[col].cat.categories) + 1  # +1 for the 0 reserved code

            # Create a COO matrix from the codes: (data, (row index, column index))
            # The data array is a 1 for every non-zero code
            row_indices = np.arange(n_rows)
            data_values = np.ones(n_rows, dtype=np.int8)

            # This creates a matrix of size N_rows x N_categories for this single column
            col_sparse = csr_matrix((data_values, (row_indices, codes)), shape=(n_rows, n_cols))
            sparse_matrix_list.append(col_sparse)

            # Keep track of the feature names for later model interpretation
            # This part is now complex and should be handled by a dedicated transformer, but
            # for memory safety, we prioritize the sparse construction.

        del X, X_numeric  # Free up memory after conversion
        gc.collect()

        # 4. Horizontally stack all sparse matrices
        print("-> Stacking sparse matrices...")
        X_sparse = hstack(sparse_matrix_list).tocsr()
        print(f"\tFinal Sparse Matrix Shape: {X_sparse.shape}")

        # 5. Train/Val Split
        print(f"-> Splitting data (test_size={test_size})...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_sparse, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 6. Apply SMOTE
        print("-> Applying SMOTE to training data...")
        smote = SMOTE(random_state=random_state)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

        # 7. Scale features
        print("-> Scaling features (using CSR matrix)...")

        # Convert all necessary data structures to CSR format (for scaling)
        X_train_sm_csr = X_train_sm.tocsr()
        X_val_csr = X_val.tocsr()

        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train_sm_csr)
        X_val_scaled = scaler.transform(X_val_csr)

        # Store for later use
        self.X_train = X_train_scaled
        self.X_val = X_val_scaled
        self.y_train = y_train_sm
        self.y_val = y_val
        self.scaler = scaler
        # Feature names are lost in this manual process; you must rebuild them later if needed.

        print("\nData preparation complete! (Sparse Memory Safe)")
        gc.collect()
        return self

    def get_train_val_data(self):
        """Returns prepared training and validation sets."""
        return self.X_train, self.X_val, self.y_train, self.y_val

    def get_data(self):
        """Returns the final processed DataFrame."""
        return self.df


# --- III. EXECUTION ---

if __name__ == '__main__':
    # Initialize the pipeline
    pipeline = FraudDetectionPipeline(
        trans_path=TRAIN_TRANS,
        id_path=TRAIN_ID
    )

    # Execute the steps sequentially
    cleaned_df = pipeline.load_and_merge() \
        .handle_missing_values() \
        .create_time_features() \
        .engineer_features() \
        .prepare_for_training() \
        .get_data()

    # Get the prepared data
    X_train, X_val, y_train, y_val = pipeline.get_train_val_data()

    print("\n" + "=" * 50)
    print("READY FOR MODEL TRAINING!")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_train fraud rate: {y_train.mean():.4f}")
    print(f"y_val fraud rate: {y_val.mean():.4f}")
    print("=" * 50)