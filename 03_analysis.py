import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import os

FILENAME_CSV = "healthcare_big_data.csv"
FILENAME_PQ = "healthcare_big_data.parquet"

def train_incremental_model():
    if os.path.exists(FILENAME_PQ):
        print(f"Loading data from {FILENAME_PQ} with Dask (Parquet)...")
        ddf = dd.read_parquet(FILENAME_PQ)
    else:
        print(f"Loading data from {FILENAME_CSV} with Dask (CSV)...")
        ddf = dd.read_csv(FILENAME_CSV)
    
    # Preprocessing
    print("Preprocessing...")
    # Convert gender to numeric
    ddf['Gender_Num'] = ddf['Gender'].map({'Male': 0, 'Female': 1}, meta=('Gender', 'int'))
    
    # Handle missing values (e.g., BMI)
    # Filling NaN with mean. Note: computing mean first.
    bmi_mean = ddf['BMI'].mean().compute()
    ddf['BMI'] = ddf['BMI'].fillna(bmi_mean)
    
    # Select features and target
    features = ['Age', 'Gender_Num', 'BMI', 'Glucose_Level', 'Blood_Pressure']
    target = 'Readmitted_Within_30_Days'
    
    # Model setup (SGD supports incremental learning)
    model = SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, tol=1e-3)
    scaler = StandardScaler()
    
    # Training loop
    start_time = time.time()
    
    # Process in partitions (blocks)
    # We'll split manually into train/test simulation by using most partitions for train
    # and leaving the last few for testing.
    
    n_partitions = ddf.npartitions
    train_partitions = int(n_partitions * 0.8)
    
    print(f"Total partitions: {n_partitions}. Training on first {train_partitions}, testing on rest.")
    
    # Training Loop
    scaler_partial_fitted = False
    
    # First pass for Scaler (Dask Incremental Scaling is tricky, we'll do manual partial_fit simulation)
    # For simplicity in this demo, we'll fit scaler on a sample or iteratively if possible.
    # Standard scaler doesn't support simple partial_fit easily without tracking mean/var.
    # Let's simple approximate scaler on a large sample (first partition)
    print("Fitting scaler on first partition...")
    first_chunk = ddf.get_partition(0).compute()
    scaler.fit(first_chunk[features])
    
    processed_count = 0
    
    for i in range(train_partitions):
        partition = ddf.get_partition(i).compute()
        X_batch = partition[features]
        y_batch = partition[target]
        
        # Scale
        X_batch_scaled = scaler.transform(X_batch)
        
        # Train
        model.partial_fit(X_batch_scaled, y_batch, classes=[0, 1])
        
        processed_count += len(y_batch)
        if i % 5 == 0:
            print(f"Processed partition {i}/{train_partitions} ({processed_count} rows)...")
            
    train_time = time.time() - start_time
    print(f"Training completed on {processed_count} rows in {train_time:.2f} s.")
    
    # Evaluation
    print("Evaluating...")
    y_true = []
    y_pred = []
    
    for i in range(train_partitions, n_partitions):
        partition = ddf.get_partition(i).compute()
        if len(partition) == 0: continue
        
        X_test_batch = partition[features]
        y_test_batch = partition[target]
        
        X_test_scaled = scaler.transform(X_test_batch)
        predictions = model.predict(X_test_scaled)
        
        y_true.extend(y_test_batch)
        y_pred.extend(predictions)
        
    acc = accuracy_score(y_true, y_pred)
    print(f"\nModel Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Save Model and Scaler
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    joblib.dump(model, 'outputs/big_data_model.pkl')
    joblib.dump(scaler, 'outputs/big_data_scaler.pkl')
    print("Model and Scaler saved to outputs/")

if __name__ == "__main__":
    if not os.path.exists(FILENAME_PQ) and not os.path.exists(FILENAME_CSV):
        print(f"Data files not found. Run 01_generate_big_data.py first.")
    else:
        train_incremental_model()
