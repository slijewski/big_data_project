import pandas as pd
import dask.dataframe as dd
import time
import os
import psutil

FILENAME = "healthcare_big_data.csv"
FILENAME_PQ = "healthcare_big_data.parquet"

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # MB

def benchmark_pandas():
    print("\n--- Pandas Benchmark (CSV) ---")
    start_mem = get_memory_usage()
    start_time = time.time()
    
    print("Loading CSV with Pandas...")
    df = pd.read_csv(FILENAME)
    load_time = time.time() - start_time
    print(f"Pandas Load Time: {load_time:.4f} s")
    
    start_calc = time.time()
    print("Calculating mean age by gender...")
    result = df.groupby('Gender')['Age'].mean()
    print(result)
    calc_time = time.time() - start_calc
    print(f"Pandas Calculation Time: {calc_time:.4f} s")
    
    end_mem = get_memory_usage()
    print(f"Pandas Memory Usage Increase: {end_mem - start_mem:.2f} MB")
    
    return load_time + calc_time

def benchmark_dask_csv():
    print("\n--- Dask Benchmark (CSV) ---")
    start_mem = get_memory_usage()
    start_time = time.time()
    
    print("Loading CSV with Dask...")
    ddf = dd.read_csv(FILENAME)
    load_time = time.time() - start_time
    print(f"Dask Load Time (Lazy): {load_time:.4f} s")
    
    start_calc = time.time()
    print("Calculating mean age by gender...")
    result = ddf.groupby('Gender')['Age'].mean().compute()
    print(result)
    
    calc_time = time.time() - start_calc
    print(f"Dask Calculation Time: {calc_time:.4f} s")
    
    end_mem = get_memory_usage()
    print(f"Dask Memory Usage Increase: {end_mem - start_mem:.2f} MB")
    
    return calc_time

def benchmark_dask_parquet():
    print("\n--- Dask Benchmark (PARQUET) ---")
    if not os.path.exists(FILENAME_PQ):
        print("Parquet file not found. Skipping...")
        return float('inf')
        
    start_mem = get_memory_usage()
    start_time = time.time()
    
    print("Loading Parquet with Dask (Column Pruning active)...")
    # Column pruning: ONLY read Gender and Age
    ddf = dd.read_parquet(FILENAME_PQ, columns=['Gender', 'Age'])
    load_time = time.time() - start_time
    print(f"Dask PQ Load Time (Lazy): {load_time:.4f} s")
    
    start_calc = time.time()
    result = ddf.groupby('Gender')['Age'].mean().compute()
    print(result)
    
    calc_time = time.time() - start_calc
    print(f"Dask PQ Calculation Time: {calc_time:.4f} s")
    
    end_mem = get_memory_usage()
    print(f"Dask PQ Memory Usage Increase: {end_mem - start_mem:.2f} MB")
    
    return calc_time

if __name__ == "__main__":
    if not os.path.exists(FILENAME):
        print(f"File {FILENAME} not found. Run 01_generate_big_data.py first.")
    else:
        file_size_csv = os.path.getsize(FILENAME) / (1024 * 1024)
        print(f"Benchmarking on file size: {file_size_csv:.2f} MB CSV")
        
        results = {}
        
        try:
            results['Pandas_CSV'] = benchmark_pandas()
        except MemoryError:
            print("Pandas failed with MemoryError!")
            results['Pandas_CSV'] = float('inf')
            
        results['Dask_CSV'] = benchmark_dask_csv()
        results['Dask_Parquet'] = benchmark_dask_parquet()
        
        print("\n--- FINAL SUMMARY ---")
        for name, duration in results.items():
            print(f"{name:15} : {duration:.4f} s")
            
        if results['Dask_Parquet'] != float('inf'):
            speedup = results['Dask_CSV'] / results['Dask_Parquet']
            print(f"\nParquet vs CSV (Dask) Speedup: {speedup:.2f}x")
            
            if results['Pandas_CSV'] != float('inf'):
                total_win = results['Pandas_CSV'] / results['Dask_Parquet']
                print(f"Total Scalability Win (Pandas CSV vs Dask PQ): {total_win:.2f}x")
