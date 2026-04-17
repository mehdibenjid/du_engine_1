import logging
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def display_embeddings(strategy, dimension=1024, save_csv=False):
    # 1. Load the generated Parquet file
    file_path = f'../artifacts/embeddings/embeddings_{strategy}_{dimension}d_DU.parquet'

    try:
        df = pd.read_parquet(file_path)

        logger.info("Successfully loaded: %s | rows=%d", file_path, len(df))

        # 2. Check for missing embeddings
        missing_count = df['vector'].isnull().sum()
        if missing_count == 0:
            logger.info("No missing embeddings found.")
        else:
            logger.warning("%d rows have missing embeddings.", missing_count)

        # 3. Inspect a sample embedding
        if len(df) > 0:
            sample_vector = df['vector'].iloc[0]

            if isinstance(sample_vector, (list, np.ndarray)):
                dim = len(sample_vector)
                logger.info("Vector Dimension: %d (Expected: %d)", dim, dimension)
                logger.debug("Sample (first 5 values): %s", sample_vector[:5])

                if dim == dimension:
                    logger.info("Dimension check passed for Amazon Titan v2.")
                else:
                    logger.warning("Unexpected dimension %d! Check model parameters.", dim)
            else:
                logger.error(
                    "Embedding column contains %s instead of list/array.",
                    type(sample_vector),
                )

        # 4. View the Data (only meaningful in interactive sessions)
        logger.debug("DataFrame head:\n%s", df.head().to_string())

    except FileNotFoundError:
        logger.error("File not found: %s. Make sure the script finished saving.", file_path)
    except Exception as e:
        logger.error("Error loading file: %s", e)

    if save_csv:
        df.to_csv(f"{strategy}_embeddings_{dimension}d.csv", sep=";", index=None)


def analyze_embeddings(strategy, dimension=1024):

    file_path = f'../artifacts/embeddings/embeddings_{strategy}_{dimension}d_DU.parquet'

    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        return

    logger.info("Loading %s...", file_path)
    df = pd.read_parquet(file_path)

    # Check if embedding column exists and is not empty
    if "vector" not in df.columns or df["vector"].isnull().all():
        logger.error("No valid embeddings found in file.")
        return

    # 1. Convert list of lists to 2D NumPy Array
    try:
        matrix = np.stack(df["vector"].values)
    except ValueError as e:
        logger.error("Error converting to matrix (possibly inconsistent dimensions): %s", e)
        return

    n_samples, n_dim = matrix.shape
    total_elements = matrix.size

    # --- 2. CALCULATE METRICS ---

    n_zeros = np.count_nonzero(matrix == 0)
    sparsity_ratio = n_zeros / total_elements

    row_norms = np.linalg.norm(matrix, axis=1)
    avg_norm = np.mean(row_norms)
    is_normalized = np.allclose(row_norms, 1.0, atol=1e-4)

    min_val = np.min(matrix)
    max_val = np.max(matrix)
    mean_val = np.mean(matrix)
    std_val = np.std(matrix)

    # --- 3. LOG REPORT ---
    logger.info(
        "EMBEDDING ANALYSIS: STRATEGY %s | %d rows x %d dims | "
        "zeros=%d sparsity=%.2f%% | norm_avg=%.4f normalized=%s | "
        "min=%.4f max=%.4f mean=%.4f std=%.4f",
        strategy, n_samples, n_dim,
        n_zeros, sparsity_ratio * 100,
        avg_norm, "YES" if is_normalized else "NO",
        min_val, max_val, mean_val, std_val,
    )

    # --- 4. VISUALIZATION (Optional) ---
    plt.figure(figsize=(12, 4))

    # Plot 1: Histogram of all values (flattened)
    plt.subplot(1, 2, 1)
    # Sample a subset if matrix is huge to speed up plotting
    sample_data = matrix.flatten()
    if len(sample_data) > 100000: 
        sample_data = np.random.choice(sample_data, 100000)
    
    sns.histplot(sample_data, bins=50, kde=True, color='skyblue')
    plt.title(f"Distribution of Values (Sampled)\nStrategy {strategy}")
    plt.xlabel("Embedding Value")
    plt.yscale('log') # Log scale helps see small tails vs huge spike at 0

    # Plot 2: Norm distribution
    plt.subplot(1, 2, 2)
    
    # Check if there is meaningful variance
    # If the variance is near-zero, we force a fixed window around the mean
    if np.std(row_norms) < 1e-6:
        center = np.mean(row_norms)
        sns.histplot(row_norms, bins=20, color='salmon', binrange=(center - 0.1, center + 0.1))
        plt.annotate("Near-zero variance\n(Vectors are normalized)", 
                     xy=(0.5, 0.5), xycoords='axes fraction', ha='center')
    else:
        sns.histplot(row_norms, bins=20, color='salmon')

    plt.title("Distribution of Vector Norms")
    plt.xlabel("L2 Norm")

    plt.tight_layout()
    plt.show()

def display_time(file_path = '../time.json'):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        # Fallback dummy data for demonstration
        data = {
        "total_row": 14227, 
        "configurations": [
            {"dimension": 1024, "A": 750, "B": 2676.61, "C": 657.86, "D": 835.27, "E": 0.75},
            {"dimension": 512, "A": 704.60, "B": 2579.69, "C": 742.10, "D": 774.95, "E": 0.56},
            {"dimension": 256, "A": 807.68, "B": 2516.31, "C": 942.07, "D": 658.53, "E": 0.46}
        ]
        }

    # 2. Create DataFrame
    df = pd.DataFrame(data["configurations"])

    # Rename strategies for the legend
    name_map = {
        "A": "Repeat (A)",
        "B": "Weights (B)",
        "C": "Concat (C)",
        "D": "Raw (D)",
        "E": "TF-IDF (E)"
    }
    # Only rename columns that actually exist in the dataframe
    df.rename(columns=name_map, inplace=True)

    # 3. Convert from Wide to Long format
    df_melted = df.melt(id_vars="dimension", var_name="Strategy", value_name="Seconds")

    # 4. CONVERT TO MINUTES
    df_melted["Minutes"] = df_melted["Seconds"] / 60

    # 5. Plotting
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 7))

    # Draw Bar Chart
    ax = sns.barplot(
        data=df_melted,
        x="dimension",
        y="Minutes",
        hue="Strategy",
        palette="viridis",
        edgecolor="black",
        linewidth=1
    )

    # Add Labels on top of bars
    for container in ax.containers:
        # Format to 2 decimal places
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=11)

    # Formatting
    # FIX: Use single quotes for the key inside the double-quoted f-string
    # Added .get() to be safe if the key is missing in the JSON
    total_rows = data.get('total_row', 'Unknown')
    plt.title(f"Execution Time by Strategy (Minutes) - Total Rows {total_rows}", fontsize=18, weight='bold', pad=20)

    plt.ylabel("Time in Minutes", fontsize=15)
    plt.xlabel("Target Dimension", fontsize=15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Strategy")

    plt.tight_layout()
    plt.show()
