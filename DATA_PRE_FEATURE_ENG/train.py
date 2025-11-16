from deep_feature_eng import FraudDetectionPipeline
from model import FraudModel
from file_path import TRAIN_TRANS, TRAIN_ID

if __name__ == '__main__':
    # PHASE 1 & 2: Data Pipeline
    print("=" * 60)
    print("PHASE 1 & 2: DATA PREPARATION")
    print("=" * 60)

    pipeline = FraudDetectionPipeline(TRAIN_TRANS, TRAIN_ID)

    pipeline.load_and_merge() \
        .handle_missing_values() \
        .create_time_features() \
        .engineer_features()\
        .prepare_for_training()

    X_train, X_val, y_train, y_val = pipeline.get_train_val_data()

    print(f"\n✅ Data Ready: Train={X_train.shape}, Val={X_val.shape}")

    # PHASE 3: Model Training
    print("\n" + "=" * 60)
    print("PHASE 3: MODEL TRAINING")
    print("=" * 60)

    model = FraudModel(
        input_dim=2196,
        hidden_dim=128,
        dropout=0.3,
        default_lr=1e-4
    )

    # Train model
    model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=10,
        batch_size=1024,
        clip_norm=1.0,
        lr=5e-5
    )

    # Evaluate
    metrics = model.evaluate(X_val, y_val, threshold=0.5)

    # Save model
    model.save('fraud_model_final.pth')


print("\n" + "=" * 60)
print("🎉 PIPELINE COMPLETE!")
print("=" * 60)