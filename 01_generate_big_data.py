import pandas as pd
import numpy as np
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_chunk(chunk_size=100_000, start_id=0):
    """Generates a chunk of synthetic health data."""
    ids = np.arange(start_id, start_id + chunk_size)
    
    age = np.random.randint(18, 90, chunk_size)
    
    bmi = np.random.normal(28, 6, chunk_size).round(1)
    
    glucose = np.random.gamma(shape=10, scale=10, size=chunk_size).astype(int) + 20
    
    bp = np.random.normal(120, 15, chunk_size).astype(int)
    
    risk_score = (glucose - 100)/50 + (bmi - 25)/10 + (age - 50)/20
    prob = 1 / (1 + np.exp(-risk_score))
    readmitted = (np.random.random(chunk_size) < prob).astype(int)
    
    gender = np.random.choice(['Male', 'Female'], chunk_size)
    
    df = pd.DataFrame({
        'Patient_ID': ids,
        'Age': age,
        'Gender': gender,
        'BMI': bmi,
        'Glucose_Level': glucose,
        'Blood_Pressure': bp,
        'Diabetes_Risk': 0,
        'Readmitted_Within_30_Days': readmitted
    })
    
    mask = np.random.random(chunk_size) < 0.05
    df.loc[mask, 'BMI'] = np.nan
    
    return df

def generate_big_data(total_rows=5_000_000, chunk_size=500_000, filename="healthcare_big_data.csv"):
    parquet_filename = filename.replace(".csv", ".parquet")
    logging.info(f"Generating {total_rows} rows of synthetic data...")
    start_time = time.time()
    
    header_written = False
    
    all_chunks = []
    
    for i in range(0, total_rows, chunk_size):
        current_chunk_size = min(chunk_size, total_rows - i)
        logging.info(f"Generating chunk {i} to {i+current_chunk_size}...")
        
        df = generate_chunk(current_chunk_size, start_id=i)
        
        mode = 'w' if not header_written else 'a'
        header = not header_written
        df.to_csv(filename, mode=mode, header=header, index=False)
        header_written = True
        
        all_chunks.append(df)
        
    logging.info("Saving to Parquet format (Partitioned)...")
    import dask.dataframe as dd_gen
    dd_final = dd_gen.from_pandas(pd.concat(all_chunks, ignore_index=True), npartitions=10)
    dd_final.to_parquet(parquet_filename, engine='pyarrow', write_index=False)
        
    end_time = time.time()
    csv_size = os.path.getsize(filename) / (1024 * 1024)
    pq_size = sum(os.path.getsize(os.path.join(parquet_filename, f)) for f in os.listdir(parquet_filename) if os.path.isfile(os.path.join(parquet_filename, f))) / (1024 * 1024)
    
    logging.info(f"\nDone! Generated {total_rows} rows in {end_time - start_time:.2f} seconds.")
    logging.info(f"CSV Size: {csv_size:.2f} MB")
    logging.info(f"Parquet Folder Size: {pq_size:.2f} MB")
    logging.info(f"Parquet compression ratio: {csv_size/pq_size:.2f}x")

if __name__ == "__main__":
    generate_big_data(total_rows=5_000_000)
