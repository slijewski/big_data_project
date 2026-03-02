import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

FILENAME_CSV = "healthcare_big_data.csv"
FILENAME_PQ = "healthcare_big_data.parquet"

def train_incremental_model():
    if os.path.exists(FILENAME_PQ):
        logging.info(f"Loading data from {FILENAME_PQ} with Dask (Parquet)...")
        ddf = dd.read_parquet(FILENAME_PQ)
    else:
        logging.info(f"Loading data from {FILENAME_CSV} with Dask (CSV)...")
        ddf = dd.read_csv(FILENAME_CSV)
    
    logging.info("Preprocessing...")
    ddf['Gender_Num'] = ddf['Gender'].map({'Male': 0, 'Female': 1}, meta=('Gender', 'int'))
    
    bmi_mean = ddf['BMI'].mean().compute()
    ddf['BMI'] = ddf['BMI'].fillna(bmi_mean)
    
    features = ['Age', 'Gender_Num', 'BMI', 'Glucose_Level', 'Blood_Pressure']
    target = 'Readmitted_Within_30_Days'
    
    model = SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, tol=1e-3)
    scaler = StandardScaler()
    
    start_time = time.time()
    
    n_partitions = ddf.npartitions
    train_partitions = int(n_partitions * 0.8)
    
    logging.info(f"Total partitions: {n_partitions}. Training on first {train_partitions}, testing on rest.")
    
    logging.info("Fitting scaler on first partition...")
    first_chunk = ddf.get_partition(0).compute()
    scaler.fit(first_chunk[features])
    
    processed_count = 0
    
    for i in range(train_partitions):
        partition = ddf.get_partition(i).compute()
        X_batch = partition[features]
        y_batch = partition[target]
        
        X_batch_scaled = scaler.transform(X_batch)
        
        model.partial_fit(X_batch_scaled, y_batch, classes=[0, 1])
        
        processed_count += len(y_batch)
        if i % 5 == 0:
            logging.info(f"Processed partition {i}/{train_partitions} ({processed_count} rows)...")
            
    train_time = time.time() - start_time
    logging.info(f"Training completed on {processed_count} rows in {train_time:.2f} s.")
    
    logging.info("Evaluating...")
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
    logging.info(f"\nModel Accuracy: {acc:.4f}")
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_true, y_pred))
    
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    joblib.dump(model, 'outputs/big_data_model.pkl')
    joblib.dump(scaler, 'outputs/big_data_scaler.pkl')
    logging.info("Model and Scaler saved to outputs/")

if __name__ == "__main__":
    if not os.path.exists(FILENAME_PQ) and not os.path.exists(FILENAME_CSV):
        logging.info(f"Data files not found. Run 01_generate_big_data.py first.")
    else:
        train_incremental_model()
