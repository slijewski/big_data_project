import pandas as pd
import dask.dataframe as dd
import time
import os
import psutil
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

FILENAME = "healthcare_big_data.csv"
FILENAME_PQ = "healthcare_big_data.parquet"

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def benchmark_pandas():
    logging.info("\n--- Pandas Benchmark (CSV) ---")
    start_mem = get_memory_usage()
    start_time = time.time()
    
    logging.info("Loading CSV with Pandas...")
    df = pd.read_csv(FILENAME)
    load_time = time.time() - start_time
    logging.info(f"Pandas Load Time: {load_time:.4f} s")
    
    start_calc = time.time()
    logging.info("Calculating mean age by gender...")
    result = df.groupby('Gender')['Age'].mean()
    logging.info(result)
    calc_time = time.time() - start_calc
    logging.info(f"Pandas Calculation Time: {calc_time:.4f} s")
    
    end_mem = get_memory_usage()
    logging.info(f"Pandas Memory Usage Increase: {end_mem - start_mem:.2f} MB")
    
    return load_time + calc_time

def benchmark_dask_csv():
    logging.info("\n--- Dask Benchmark (CSV) ---")
    start_mem = get_memory_usage()
    start_time = time.time()
    
    logging.info("Loading CSV with Dask...")
    ddf = dd.read_csv(FILENAME)
    load_time = time.time() - start_time
    logging.info(f"Dask Load Time (Lazy): {load_time:.4f} s")
    
    start_calc = time.time()
    logging.info("Calculating mean age by gender...")
    result = ddf.groupby('Gender')['Age'].mean().compute()
    logging.info(result)
    
    calc_time = time.time() - start_calc
    logging.info(f"Dask Calculation Time: {calc_time:.4f} s")
    
    end_mem = get_memory_usage()
    logging.info(f"Dask Memory Usage Increase: {end_mem - start_mem:.2f} MB")
    
    return calc_time

def benchmark_dask_parquet():
    logging.info("\n--- Dask Benchmark (PARQUET) ---")
    if not os.path.exists(FILENAME_PQ):
        logging.info("Parquet file not found. Skipping...")
        return float('inf')
        
    start_mem = get_memory_usage()
    start_time = time.time()
    
    logging.info("Loading Parquet with Dask (Column Pruning active)...")
    ddf = dd.read_parquet(FILENAME_PQ, columns=['Gender', 'Age'])
    load_time = time.time() - start_time
    logging.info(f"Dask PQ Load Time (Lazy): {load_time:.4f} s")
    
    start_calc = time.time()
    result = ddf.groupby('Gender')['Age'].mean().compute()
    logging.info(result)
    
    calc_time = time.time() - start_calc
    logging.info(f"Dask PQ Calculation Time: {calc_time:.4f} s")
    
    end_mem = get_memory_usage()
    logging.info(f"Dask PQ Memory Usage Increase: {end_mem - start_mem:.2f} MB")
    
    return calc_time

if __name__ == "__main__":
    if not os.path.exists(FILENAME):
        logging.info(f"File {FILENAME} not found. Run 01_generate_big_data.py first.")
    else:
        file_size_csv = os.path.getsize(FILENAME) / (1024 * 1024)
        logging.info(f"Benchmarking on file size: {file_size_csv:.2f} MB CSV")
        
        results = {}
        
        try:
            results['Pandas_CSV'] = benchmark_pandas()
        except MemoryError:
            logging.info("Pandas failed with MemoryError!")
            results['Pandas_CSV'] = float('inf')
            
        results['Dask_CSV'] = benchmark_dask_csv()
        results['Dask_Parquet'] = benchmark_dask_parquet()
        
        logging.info("\n--- FINAL SUMMARY ---")
        for name, duration in results.items():
            logging.info(f"{name:15} : {duration:.4f} s")
            
        if results['Dask_Parquet'] != float('inf'):
            speedup = results['Dask_CSV'] / results['Dask_Parquet']
            logging.info(f"\nParquet vs CSV (Dask) Speedup: {speedup:.2f}x")
            
            if results['Pandas_CSV'] != float('inf'):
                total_win = results['Pandas_CSV'] / results['Dask_Parquet']
                logging.info(f"Total Scalability Win (Pandas CSV vs Dask PQ): {total_win:.2f}x")
