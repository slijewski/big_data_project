# 🏥 Scalable Healthcare Analytics: Big Data & Incremental Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Dask](https://img.shields.io/badge/Interface-Dask-4D90FE.svg)](https://dask.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

## Overview

Modern healthcare datasets often exceed available system memory (RAM), necessitating specialized architectures for data processing and machine learning. This repository demonstrates a scalable pipeline for **Healthcare Big Data Analytics**, processing millions of records through a combination of out-of-core computing and incremental model training.

The project simulates a predictive clinical environment where patient hospital readmission risk is analyzed using a dataset of **5 million records**, optimized for performance and memory efficiency.

## 🏗️ Technical Architecture

### 1. High-Performance Data Storage (Parquet)
Traditional CSV files are inefficient for large-scale analytics due to their row-major format and lack of schema. This project utilizes the **Apache Parquet** format:
- **Columnar Storage**: Enables efficient column pruning (only reading necessary features).
- **Partitioning**: The 5M-row dataset is split into optimized partitions for parallel processing.
- **Compression**: Achieves significant storage savings compared to flat CSV files.

### 2. Out-of-Core Processing (Dask)
To handle data larger than RAM, **Dask** is employed as the primary engine. It enables:
- **Lazy Evaluation**: Computation graphs are built first and executed only when needed.
- **Parallelism**: Efficient use of multi-core CPUs for data transformation and aggregation.

### 3. Incremental Machine Learning (Online Learning)
Instead of loading the entire dataset into memory for training, we use an **Online Learning** approach with Scikit-learn's `SGDClassifier`:
- **Stochastic Gradient Descent (SGD)**: The model is updated iteratively using small partitions (blocks) of data.
- **Memory Efficiency**: Constant memory footprint regardless of total dataset size.
- **Scalability**: Capable of training on datasets that scale into hundreds of millions of rows.

## 📊 Performance Benchmarking

The `02_benchmark.py` script compares traditional Pandas against the Dask/Parquet architecture. Typical findings on this scale include:
- **Load Time**: Parquet is significantly faster than CSV due to metadata indexing.
- **Memory Usage**: Dask maintains a low, stable memory profile, while Pandas risk `MemoryError` on 8GB/16GB machines when processing millions of rows.

## 🖥️ Predictive Dashboard

The Streamlit application (`app.py`) provides an interface for:
- **Real-time Prediction**: Estimating patient readmission risk based on clinical features (Age, BMI, Glucose, etc.).
- **Scalable Insights**: Visualizing population-level metrics calculated on-the-fly across the massive dataset using Dask.

## 🚀 Getting Started

### Installation

1. Install required dependencies:
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

2. Generate the synthetic Big Data (approx. 5 million rows):
   ```bash
   python 01_generate_big_data.py
   ```

3. Run benchmarks to compare performance:
   ```bash
   python 02_benchmark.py
   ```

4. Train the incremental model:
   ```bash
   python 03_analysis.py
   ```

5. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

## 📁 Repository Structure

```text
├── .python-version           # Python version pin (uv)
├── 01_generate_big_data.py   # Synthetic data generation (CSV/Parquet)
├── 02_benchmark.py           # Pandas vs Dask performance analysis
├── 03_analysis.py            # Incremental training pipeline (SGD)
├── app.py                    # Streamlit analytics dashboard
├── requirements.txt          # Project dependencies
├── uv.lock                   # Lockfile for reproducible environment
└── outputs/                  # Serialized models and scalers
```

## 📜 Disclaimer

> **DATA NOTICE:** This project uses **purely synthetic data** generated for demonstration purposes. It does not reflect real medical registry data and should not be used for clinical decision support or research.

---

## 👨‍🔬Author

Sebastian Lijewski
PhD in Pharmaceutical Sciences
