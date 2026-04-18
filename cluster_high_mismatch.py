#!/usr/bin/env python3
"""
Cluster problems by embeddings and identify high-mismatch clusters.
Filters 250k dataset to focus on problematic clusters for re-prediction.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set
import numpy as np
from tqdm import tqdm

# Try importing required libraries
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    os.system("pip install sentence-transformers -q")
    from sentence_transformers import SentenceTransformer

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
except ImportError:
    print("Installing scikit-learn...")
    os.system("pip install scikit-learn -q")
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

try:
    import matplotlib.pyplot as plt
    import pickle
except ImportError:
    print("Installing matplotlib and pickle...")
    os.system("pip install matplotlib -q")
    import matplotlib.pyplot as plt
    import pickle

# Configuration
LOCAL_DATA_DIR = Path("/home/malam/wsl-tunix/imo/openmath_data")
PREDICTIONS_LOG_PATH = LOCAL_DATA_DIR / "predictions_log.jsonl"
FILTERED_DATASET_PATH = LOCAL_DATA_DIR / "openmath_filtered_integers.jsonl"
OUTPUT_PATH = LOCAL_DATA_DIR / "high_mismatch_clusters.jsonl"

# Embedding model (using a fast, general-purpose model)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, lightweight, good for semantic similarity
N_CLUSTERS = 15  # Number of clusters for k-means
HIGH_MISMATCH_THRESHOLD = 0.59  # Consider clusters with >50% mismatch rate as "high mismatch"

# Paths for saved models and stats
KMEANS_MODEL_PATH = LOCAL_DATA_DIR / "kmeans_model.pkl"
CLUSTER_STATS_PATH = LOCAL_DATA_DIR / "cluster_stats.json"
ALL_EXAMPLES_WITH_CLUSTERS_PATH = LOCAL_DATA_DIR / "all_examples_with_clusters.jsonl"
MISMATCH_RATE_DISTRIBUTION_PLOT_PATH = LOCAL_DATA_DIR / "mismatch_rate_distribution.png"


def load_predictions_log(predictions_log_path: Path) -> Dict:
    """Load predictions log and extract problems with mismatch info."""
    print(f"\n[Step 1] Loading predictions log from {predictions_log_path}...")
    
    if not predictions_log_path.exists():
        raise FileNotFoundError(f"Predictions log not found: {predictions_log_path}")
    
    predictions = []
    logged_indices = set()
    
    with open(predictions_log_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading predictions log"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                idx = entry.get("idx")
                problem = entry.get("problem", "")
                score_match = entry.get("score_match", "")
                predicted_answer = entry.get("predicted_answer")
                expected_answer = entry.get("expected_answer")
                
                if idx is not None and problem:
                    predictions.append({
                        "idx": idx,
                        "problem": problem,
                        "score_match": score_match,
                        "predicted_answer": predicted_answer,
                        "expected_answer": expected_answer,
                        "is_mismatch": score_match.lower() != "match"
                    })
                    logged_indices.add(idx)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"⚠ Warning: Skipping malformed line: {e}")
                continue
    
    print(f"✓ Loaded {len(predictions)} predictions from log")
    print(f"  Logged indices: {len(logged_indices)} unique indices")
    
    return {
        "predictions": predictions,
        "logged_indices": logged_indices
    }


def create_elbow_plot(embeddings: np.ndarray, k_range: range = range(5, 101, 5), 
                      output_path: Path = None) -> Dict[int, float]:
    """Create elbow plot to determine optimal number of clusters.
    
    Computes within-cluster sum of squares (WCSS/inertia) for different k values.
    
    Args:
        embeddings: Problem embeddings array
        k_range: Range of k values to test
        output_path: Optional path to save the plot. If None, saves to LOCAL_DATA_DIR.
    
    Returns:
        Dictionary mapping k values to their inertia scores
    """
    print(f"\n[Elbow Plot] Testing {len(k_range)} different k values...")
    
    if output_path is None:
        output_path = LOCAL_DATA_DIR / "kmeans_elbow_plot.png"
    
    inertias = {}
    
    # Test each k value
    for k in tqdm(k_range, desc="Computing inertias"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100, verbose=0)
        kmeans.fit(embeddings)
        inertias[k] = kmeans.inertia_
    
    # Create the plot
    print(f"  Creating elbow plot...")
    plt.figure(figsize=(10, 6))
    k_values = sorted(inertias.keys())
    inertia_values = [inertias[k] for k in k_values]
    
    plt.plot(k_values, inertia_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Within-Cluster Sum of Squares (WCSS / Inertia)', fontsize=12)
    plt.title('K-Means Elbow Plot\n(Find the "elbow" point to determine optimal k)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values[::2] if len(k_values) > 10 else k_values)  # Show every other tick if many values
    
    # Add annotations for key points
    plt.tight_layout()
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Elbow plot saved to: {output_path}")
    
    # Print some statistics
    print(f"\n  Inertia values:")
    print(f"    k={k_values[0]:3d}: {inertia_values[0]:.2e}")
    print(f"    k={k_values[-1]:3d}: {inertia_values[-1]:.2e}")
    print(f"    Decrease: {((inertia_values[0] - inertia_values[-1]) / inertia_values[0] * 100):.1f}%")
    
    # Calculate rate of change (second derivative approximation)
    if len(k_values) > 1:
        rates_of_change = []
        for i in range(1, len(k_values)):
            rate = (inertia_values[i-1] - inertia_values[i]) / (k_values[i] - k_values[i-1])
            rates_of_change.append((k_values[i], rate))
        
        # Find where the rate of change starts to decrease significantly (elbow)
        # This is a simple heuristic: find where consecutive rate changes start decreasing
        if len(rates_of_change) > 2:
            rate_changes = []
            for i in range(1, len(rates_of_change)):
                change = rates_of_change[i][1] - rates_of_change[i-1][1]
                rate_changes.append((rates_of_change[i][0], change))
            
            # Suggest elbow point (where rate of decrease slows)
            # Find the first point where rate of change becomes less negative
            suggested_k = None
            for k, change in rate_changes:
                if change > 0:  # Rate of decrease is slowing
                    suggested_k = k
                    break
            
            if suggested_k:
                print(f"  ⚡ Suggested elbow point (k={suggested_k}) based on rate of change")
    
    plt.close()  # Close to free memory
    
    return inertias


def embed_problems(predictions: List[Dict], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """Generate embeddings for problems using sentence-transformers."""
    print(f"\n[Step 2] Generating embeddings using model: {model_name}...")
    
    # Initialize model
    print(f"  Loading embedding model...")
    model = SentenceTransformer(model_name)
    
    # Extract problem texts
    problems = [pred["problem"] for pred in tqdm(predictions, desc="Extracting problems")]
    
    # Generate embeddings (model handles batching internally)
    print(f"  Generating embeddings for {len(problems)} problems...")
    embeddings = model.encode(problems, show_progress_bar=True, batch_size=32)
    
    print(f"✓ Generated embeddings: shape {embeddings.shape}")
    return embeddings


def cluster_problems(embeddings: np.ndarray, n_clusters: int = N_CLUSTERS):
    """Perform k-means clustering on problem embeddings.
    
    Returns:
        tuple: (cluster_labels, kmeans_model)
    """
    print(f"\n[Step 3] Performing k-means clustering with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    print(f"✓ Clustering complete: {len(np.unique(cluster_labels))} unique clusters")
    
    # Save the k-means model
    try:
        KMEANS_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(KMEANS_MODEL_PATH, 'wb') as f:
            pickle.dump(kmeans, f)
        print(f"✓ Saved k-means model to: {KMEANS_MODEL_PATH}")
    except Exception as e:
        print(f"⚠ Warning: Could not save k-means model: {e}")
    
    return cluster_labels, kmeans


def calculate_cluster_mismatch_rates(predictions: List[Dict], cluster_labels: np.ndarray) -> Dict[int, Dict]:
    """Calculate mismatch rate for each cluster."""
    print(f"\n[Step 4] Calculating mismatch rates per cluster...")
    
    # Group predictions by cluster
    cluster_data = defaultdict(lambda: {"total": 0, "mismatches": 0, "indices": []})
    
    for pred, cluster_id in zip(predictions, cluster_labels):
        cluster_data[cluster_id]["total"] += 1
        cluster_data[cluster_id]["indices"].append(pred["idx"])
        if pred["is_mismatch"]:
            cluster_data[cluster_id]["mismatches"] += 1
    
    # Calculate mismatch rates
    cluster_stats = {}
    for cluster_id, data in cluster_data.items():
        mismatch_rate = data["mismatches"] / data["total"] if data["total"] > 0 else 0.0
        cluster_stats[cluster_id] = {
            "mismatch_rate": mismatch_rate,
            "total_examples": data["total"],
            "mismatches": data["mismatches"],
            "matches": data["total"] - data["mismatches"],
            "indices": data["indices"]
        }
    
    # Sort by mismatch rate (descending)
    sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]["mismatch_rate"], reverse=True)
    
    print(f"\n  Cluster mismatch rate summary (top 10):")
    for cluster_id, stats in sorted_clusters[:10]:
        print(f"    Cluster {cluster_id:2d}: {stats['mismatch_rate']:.2%} "
              f"({stats['mismatches']}/{stats['total_examples']} mismatches)")
    
    # Save cluster stats (removing indices list to keep file size manageable)
    stats_to_save = {
        cluster_id: {
            "mismatch_rate": stats["mismatch_rate"],
            "total_examples": stats["total_examples"],
            "mismatches": stats["mismatches"],
            "matches": stats["matches"]
        }
        for cluster_id, stats in cluster_stats.items()
    }
    try:
        CLUSTER_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CLUSTER_STATS_PATH, 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, indent=2)
        print(f"✓ Saved cluster stats to: {CLUSTER_STATS_PATH}")
    except Exception as e:
        print(f"⚠ Warning: Could not save cluster stats: {e}")
    
    return cluster_stats


def load_all_filtered_examples(filtered_dataset_path: Path) -> List[Dict]:
    """Load all 250k filtered examples from JSONL."""
    print(f"\n[Step 5] Loading all filtered examples from {filtered_dataset_path}...")
    
    if not filtered_dataset_path.exists():
        raise FileNotFoundError(f"Filtered dataset not found: {filtered_dataset_path}")
    
    examples = []
    with open(filtered_dataset_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading filtered examples"):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                examples.append(example)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠ Warning: Skipping malformed line: {e}")
                continue
    
    print(f"✓ Loaded {len(examples)} examples from filtered dataset")
    return examples


def assign_clusters_to_all_examples(examples: List[Dict], kmeans_model, 
                                     model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """Assign clusters to all 250k examples using the trained k-means model from predictions log.
    
    This ensures cluster IDs match between predictions log and all examples.
    """
    print(f"\n[Step 6] Assigning clusters to all {len(examples)} examples...")
    
    # Load embedding model
    print(f"  Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Extract problem texts
    all_problems = []
    for ex in tqdm(examples, desc="Extracting problems"):
        problem = ex.get("problem", "")
        if not problem:
            # Try alternative field names
            for field in ["question", "text", "input"]:
                if field in ex:
                    problem = str(ex[field])
                    break
        all_problems.append(problem)
    
    # Generate embeddings for all examples
    print(f"  Generating embeddings for {len(all_problems)} problems...")
    all_embeddings = model.encode(all_problems, show_progress_bar=True, batch_size=32)
    
    # Use the trained k-means model to predict clusters (this ensures cluster IDs match)
    print(f"  Assigning clusters using trained k-means model...")
    all_cluster_labels = kmeans_model.predict(all_embeddings)
    
    print(f"✓ Assigned clusters: {len(np.unique(all_cluster_labels))} unique clusters")
    return all_cluster_labels


def save_all_examples_with_clusters(examples: List[Dict], cluster_labels: np.ndarray,
                                     cluster_stats: Dict[int, Dict],
                                     logged_indices: Set[int],
                                     output_path: Path = None):
    """Save all examples with their cluster assignments and mismatch rates.
    
    This saves cluster_id and cluster_mismatch_rate for every record in the dataset.
    """
    if output_path is None:
        output_path = ALL_EXAMPLES_WITH_CLUSTERS_PATH
    
    print(f"\n[Step 6.5] Saving all examples with cluster assignments to {output_path}...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex, cluster_id in tqdm(zip(examples, cluster_labels), 
                                  desc="Saving examples with clusters", 
                                  total=len(examples)):
            # Create a copy of the example
            ex_copy = dict(ex)
            
            # Add cluster info
            ex_copy["cluster_id"] = int(cluster_id)
            # Get cluster stats if available, otherwise use default
            cluster_stat = cluster_stats.get(cluster_id, {"mismatch_rate": 0.0})
            ex_copy["cluster_mismatch_rate"] = float(cluster_stat["mismatch_rate"])
            
            # Add indicator for whether this example is in predictions_log
            idx = ex.get("idx")
            ex_copy["in_predictions_log"] = (idx is not None and idx in logged_indices)
            
            f.write(json.dumps(ex_copy, ensure_ascii=False) + "\n")
            saved_count += 1
    
    print(f"✓ Saved {saved_count} examples with cluster assignments to {output_path}")


def plot_mismatch_rate_distribution(cluster_stats: Dict[int, Dict],
                                     output_path: Path = None):
    """Plot the distribution of mismatch rates across clusters.
    
    Creates a histogram showing how many clusters fall into different mismatch rate ranges.
    """
    if output_path is None:
        output_path = MISMATCH_RATE_DISTRIBUTION_PLOT_PATH
    
    print(f"\n[Visualization] Creating mismatch rate distribution plot...")
    
    # Extract mismatch rates
    mismatch_rates = [stats["mismatch_rate"] for stats in cluster_stats.values()]
    
    if len(mismatch_rates) == 0:
        print("  ⚠ No mismatch rates to plot")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Create histogram
    n_bins = min(20, len(mismatch_rates))  # Use up to 20 bins
    counts, bins, patches = plt.hist(mismatch_rates, bins=n_bins, edgecolor='black', alpha=0.7)
    
    # Color bars by mismatch rate (red for high, green for low)
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=min(mismatch_rates), vmax=max(mismatch_rates))
    for count, patch, bin_val in zip(counts, patches, bins[:-1]):
        patch.set_facecolor(cmap(norm(bin_val)))
    
    plt.xlabel('Cluster Mismatch Rate', fontsize=12)
    plt.ylabel('Number of Clusters', fontsize=12)
    plt.title(f'Distribution of Mismatch Rates Across Clusters (k={N_CLUSTERS})', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add vertical line at 0.60 threshold
    threshold = 0.60
    max_rate = max(mismatch_rates)
    if max_rate >= threshold:
        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {threshold:.1%}')
    else:
        plt.axvline(x=max_rate, color='orange', linestyle='--', linewidth=2, 
                   label=f'Max Rate (used as threshold): {max_rate:.1%}')
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Mismatch rate distribution plot saved to: {output_path}")
    
    # Print statistics
    print(f"\n  Mismatch Rate Statistics:")
    print(f"    Min: {min(mismatch_rates):.2%}")
    print(f"    Max: {max(mismatch_rates):.2%}")
    print(f"    Mean: {np.mean(mismatch_rates):.2%}")
    print(f"    Median: {np.median(mismatch_rates):.2%}")
    print(f"    Clusters >= 0.60: {sum(1 for r in mismatch_rates if r >= 0.60)}")
    
    plt.close()


def filter_high_mismatch_clusters(examples: List[Dict], cluster_labels: np.ndarray,
                                   cluster_stats: Dict[int, Dict],
                                   logged_indices: Set[int],
                                   threshold: float = 0.60) -> List[Dict]:
    """Filter examples from high-mismatch clusters and add indicator column.
    
    Uses threshold of 0.60, or the maximum mismatch rate if no cluster >= 0.60.
    """
    # Determine effective threshold
    max_mismatch_rate = max((stats["mismatch_rate"] for stats in cluster_stats.values()), default=0.0)
    effective_threshold = threshold if max_mismatch_rate >= threshold else max_mismatch_rate
    
    print(f"\n[Step 7] Filtering high-mismatch clusters...")
    print(f"  Requested threshold: {threshold:.1%}")
    print(f"  Max mismatch rate: {max_mismatch_rate:.1%}")
    print(f"  Effective threshold: {effective_threshold:.1%}")
    
    # Identify high-mismatch cluster IDs
    # Only consider clusters that have stats (appeared in predictions log)
    high_mismatch_clusters = {
        cluster_id for cluster_id, stats in cluster_stats.items()
        if stats["mismatch_rate"] >= effective_threshold
    }
    
    if len(high_mismatch_clusters) == 0:
        print(f"  ⚠ No clusters found with mismatch rate >= {effective_threshold:.1%}")
        return []
    
    print(f"  Found {len(high_mismatch_clusters)} high-mismatch clusters:")
    for cluster_id in sorted(high_mismatch_clusters):
        stats = cluster_stats[cluster_id]
        print(f"    Cluster {cluster_id:2d}: {stats['mismatch_rate']:.2%} "
              f"({stats['mismatches']}/{stats['total_examples']} mismatches)")
    
    # Filter examples from high-mismatch clusters
    filtered_examples = []
    for ex, cluster_id in tqdm(zip(examples, cluster_labels), 
                                desc="Filtering examples", 
                                total=len(examples)):
        if cluster_id in high_mismatch_clusters:
            # Create a copy of the example
            filtered_ex = dict(ex)
            
            # Add cluster info
            filtered_ex["cluster_id"] = int(cluster_id)
            # Get cluster stats if available, otherwise use default
            cluster_stat = cluster_stats.get(cluster_id, {"mismatch_rate": 0.0})
            filtered_ex["cluster_mismatch_rate"] = float(cluster_stat["mismatch_rate"])
            
            # Add indicator for whether this example is in predictions_log
            idx = ex.get("idx")
            filtered_ex["in_predictions_log"] = (idx is not None and idx in logged_indices)
            
            filtered_examples.append(filtered_ex)
    
    print(f"✓ Filtered {len(filtered_examples)} examples from high-mismatch clusters")
    
    return filtered_examples


def save_filtered_examples(filtered_examples: List[Dict], output_path: Path):
    """Save filtered examples to JSONL file."""
    print(f"\n[Step 8] Saving {len(filtered_examples)} filtered examples to {output_path}...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in tqdm(filtered_examples, desc="Writing examples"):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved to {output_path}")
    
    # Print summary statistics
    in_log_count = sum(1 for ex in filtered_examples if ex.get("in_predictions_log", False))
    not_in_log_count = len(filtered_examples) - in_log_count
    
    print(f"\n  Summary:")
    print(f"    Total filtered examples: {len(filtered_examples)}")
    print(f"    In predictions_log: {in_log_count}")
    print(f"    Not in predictions_log: {not_in_log_count}")


def plot_cluster_distribution_pca(embeddings: np.ndarray, cluster_labels: np.ndarray,
                                   cluster_stats: Dict[int, Dict],
                                   output_path: Path = None):
    """Plot cluster distribution on 2D PCA colored by mismatch rate.
    
    Args:
        embeddings: Problem embeddings array
        cluster_labels: Cluster assignments for each embedding
        cluster_stats: Dictionary with mismatch rates per cluster
        output_path: Optional path to save the plot
    """
    print(f"\n[Visualization] Creating PCA plot of cluster distribution...")
    
    if output_path is None:
        output_path = LOCAL_DATA_DIR / "cluster_pca_distribution.png"
    
    # Reduce to 2D using PCA
    print(f"  Reducing dimensions to 2D using PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Get mismatch rates for each point
    mismatch_rates = np.array([cluster_stats.get(cid, {}).get("mismatch_rate", 0.0) 
                               for cid in cluster_labels])
    
    # Create the plot
    print(f"  Creating scatter plot...")
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot colored by mismatch rate
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=mismatch_rates, cmap='RdYlGn_r',  # Red=high mismatch, Green=low mismatch
                         s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
    
    plt.colorbar(scatter, label='Cluster Mismatch Rate (%)')
    plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.1%})', 
               fontsize=12)
    plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.1%})', 
               fontsize=12)
    plt.title(f'Cluster Distribution on 2D PCA\n(Colored by Cluster Mismatch Rate, k={N_CLUSTERS})', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add cluster centers
    unique_clusters = np.unique(cluster_labels)
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_points_2d = embeddings_2d[mask]
        if len(cluster_points_2d) > 0:
            center = cluster_points_2d.mean(axis=0)
            mismatch_rate = cluster_stats.get(cluster_id, {}).get("mismatch_rate", 0.0)
            plt.plot(center[0], center[1], 'kx', markersize=15, markeredgewidth=3)
            plt.text(center[0], center[1], f'  C{cluster_id}\n  {mismatch_rate:.1%}', 
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ PCA plot saved to: {output_path}")
    
    # Print statistics
    print(f"\n  PCA Explained Variance:")
    print(f"    PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"    PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"    Total: {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]):.2%}")
    
    # Print mismatch rate distribution across clusters
    print(f"\n  Mismatch Rate Distribution by Cluster:")
    sorted_clusters_by_rate = sorted(cluster_stats.items(), 
                                     key=lambda x: x[1]["mismatch_rate"], reverse=True)
    for cluster_id, stats in sorted_clusters_by_rate:
        print(f"    Cluster {cluster_id:2d}: {stats['mismatch_rate']:.2%} "
              f"({stats['mismatches']}/{stats['total_examples']} mismatches, "
              f"{stats['total_examples']} total)")
    
    plt.close()


def plot_prediction_cluster_distribution(predictions_log_path: Path = None,
                                         output_path: Path = None):
    """Plot cluster distribution from predictions_log.jsonl using PCA.
    
    This function visualizes the clusters assigned during prediction,
    colored by cluster_mismatch_rate. If predictions don't have cluster
    assignments, it will retroactively assign them using saved models.
    
    Args:
        predictions_log_path: Path to predictions_log.jsonl. If None, uses PREDICTIONS_LOG_PATH.
        output_path: Optional path to save the plot. If None, saves to LOCAL_DATA_DIR.
    """
    if predictions_log_path is None:
        predictions_log_path = PREDICTIONS_LOG_PATH
    
    if output_path is None:
        output_path = LOCAL_DATA_DIR / "prediction_cluster_pca_distribution.png"
    
    print(f"\n[Visualization] Creating PCA plot of prediction cluster distribution...")
    print(f"  Loading predictions from: {predictions_log_path}")
    
    # Load predictions from log
    all_predictions = []
    predictions_with_clusters = 0
    with open(predictions_log_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading predictions"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                problem = entry.get("problem", "")
                if problem:
                    all_predictions.append(entry)
                    if entry.get("predicted_cluster") is not None:
                        predictions_with_clusters += 1
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                continue
    
    if len(all_predictions) == 0:
        print(f"⚠ Warning: No predictions found in {predictions_log_path}")
        return
    
    print(f"✓ Loaded {len(all_predictions)} total predictions")
    
    # Check if we need to retroactively assign clusters
    if predictions_with_clusters == 0:
        print(f"  No cluster assignments found. Attempting to assign clusters retroactively...")
        
        # Try to load clustering models
        kmeans_model_path = LOCAL_DATA_DIR / "kmeans_model.pkl"
        cluster_stats_path = LOCAL_DATA_DIR / "cluster_stats.json"
        
        if not kmeans_model_path.exists() or not cluster_stats_path.exists():
            print(f"  ⚠ Clustering models not found. Cannot assign clusters.")
            print(f"    Required files:")
            print(f"      - {kmeans_model_path}")
            print(f"      - {cluster_stats_path}")
            print(f"    Please run cluster_high_mismatch.py first to train the models.")
            return
        
        # Load models
        try:
            import pickle
            with open(kmeans_model_path, 'rb') as f:
                kmeans_model = pickle.load(f)
            with open(cluster_stats_path, 'r', encoding='utf-8') as f:
                cluster_stats = json.load(f)
            
            print(f"  ✓ Loaded clustering models. Assigning clusters to {len(all_predictions)} predictions...")
            
            # Generate embeddings and assign clusters
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            problems = [pred.get("problem", "") for pred in all_predictions]
            embeddings = embedding_model.encode(problems, show_progress_bar=True, batch_size=32)
            cluster_labels = kmeans_model.predict(embeddings)
            
            # Create predictions_data with cluster assignments
            predictions_data = []
            for pred, cluster_id in zip(all_predictions, cluster_labels):
                # Convert cluster_id to int and get mismatch rate
                cluster_id = int(cluster_id)
                cluster_stat = cluster_stats.get(str(cluster_id), cluster_stats.get(cluster_id, {}))
                mismatch_rate = float(cluster_stat.get("mismatch_rate", 0.0))
                
                predictions_data.append({
                    "problem": pred.get("problem", ""),
                    "predicted_cluster": cluster_id,
                    "cluster_mismatch_rate": mismatch_rate
                })
            
            print(f"  ✓ Assigned clusters to all predictions")
        except Exception as e:
            print(f"  ⚠ Error assigning clusters: {e}")
            return
    else:
        # Use existing cluster assignments
        print(f"  Found {predictions_with_clusters} predictions with cluster assignments")
        predictions_data = []
        for entry in all_predictions:
            problem = entry.get("problem", "")
            predicted_cluster = entry.get("predicted_cluster")
            cluster_mismatch_rate = entry.get("cluster_mismatch_rate")
            
            if problem and predicted_cluster is not None:
                predictions_data.append({
                    "problem": problem,
                    "predicted_cluster": predicted_cluster,
                    "cluster_mismatch_rate": cluster_mismatch_rate if cluster_mismatch_rate is not None else 0.0
                })
    
    if len(predictions_data) == 0:
        print(f"⚠ Warning: No predictions with cluster assignments found")
        return
    
    print(f"✓ Processing {len(predictions_data)} predictions with cluster assignments")
    
    # Extract problems and cluster info
    problems = [pred["problem"] for pred in predictions_data]
    cluster_labels = np.array([pred["predicted_cluster"] for pred in predictions_data])
    mismatch_rates = np.array([pred["cluster_mismatch_rate"] for pred in predictions_data])
    
    # Generate embeddings
    print(f"  Generating embeddings for {len(problems)} problems...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedding_model.encode(problems, show_progress_bar=True, batch_size=32)
    
    # Reduce to 2D using PCA
    print(f"  Reducing dimensions to 2D using PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create the plot
    print(f"  Creating scatter plot...")
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot colored by mismatch rate
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=mismatch_rates, cmap='RdYlGn_r',  # Red=high mismatch, Green=low mismatch
                         s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
    
    plt.colorbar(scatter, label='Cluster Mismatch Rate (%)')
    plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.1%})', 
               fontsize=12)
    plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.1%})', 
               fontsize=12)
    plt.title(f'Prediction Cluster Distribution on 2D PCA\n(Colored by Cluster Mismatch Rate, k={N_CLUSTERS})', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add cluster centers and labels
    unique_clusters = np.unique(cluster_labels)
    # Group by cluster to get mismatch rates
    cluster_mismatch_map = {}
    for pred in predictions_data:
        cluster_id = pred["predicted_cluster"]
        if cluster_id not in cluster_mismatch_map:
            cluster_mismatch_map[cluster_id] = pred["cluster_mismatch_rate"]
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_points_2d = embeddings_2d[mask]
        if len(cluster_points_2d) > 0:
            center = cluster_points_2d.mean(axis=0)
            mismatch_rate = cluster_mismatch_map.get(cluster_id, 0.0)
            plt.plot(center[0], center[1], 'kx', markersize=15, markeredgewidth=3)
            plt.text(center[0], center[1], f'  C{cluster_id}\n  {mismatch_rate:.1%}', 
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ PCA plot saved to: {output_path}")
    
    # Print statistics
    print(f"\n  PCA Explained Variance:")
    print(f"    PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"    PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"    Total: {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]):.2%}")
    
    # Print cluster distribution statistics
    print(f"\n  Cluster Distribution in Predictions:")
    from collections import Counter
    cluster_counts = Counter(cluster_labels)
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    for cluster_id, count in sorted_clusters:
        mismatch_rate = cluster_mismatch_map.get(cluster_id, 0.0)
        pct = (count / len(predictions_data)) * 100
        print(f"    Cluster {cluster_id:2d}: {count:4d} predictions ({pct:5.2f}%), "
              f"mismatch rate: {mismatch_rate:.2%}")
    
    plt.close()


def main():
    """Main execution function."""
    print("="*80)
    print("CLUSTERING HIGH-MISMATCH PROBLEMS")
    print("="*80)
    
    # Step 1: Load predictions log
    predictions_data = load_predictions_log(PREDICTIONS_LOG_PATH)
    predictions = predictions_data["predictions"]
    logged_indices = predictions_data["logged_indices"]
    
    if len(predictions) == 0:
        print("✗ No predictions found in log. Cannot proceed with clustering.")
        return
    
    # Step 2: Generate embeddings for predictions log problems
    embeddings = embed_problems(predictions)
    
    # Step 2.5: Create elbow plot to determine optimal k
    print(f"\n{'='*80}")
    print("ELBOW PLOT ANALYSIS")
    print(f"{'='*80}")
    inertias = create_elbow_plot(embeddings, k_range=range(5, 101, 5))
    print(f"\n  Review the elbow plot at: {LOCAL_DATA_DIR / 'kmeans_elbow_plot.png'}")
    print(f"  Current N_CLUSTERS setting: {N_CLUSTERS}")
    print(f"  You can adjust N_CLUSTERS in the script based on the elbow plot.\n")
    
    # Step 3: Cluster predictions log problems
    cluster_labels, kmeans_model = cluster_problems(embeddings)
    
    # Step 4: Calculate mismatch rates per cluster
    cluster_stats = calculate_cluster_mismatch_rates(predictions, cluster_labels)
    
    # Step 4.5: Create PCA visualization of cluster distribution
    plot_cluster_distribution_pca(embeddings, cluster_labels, cluster_stats)
    
    # Step 5: Load all filtered examples (250k)
    all_examples = load_all_filtered_examples(FILTERED_DATASET_PATH)
    
    # Step 6: Assign clusters to all examples using the trained k-means model
    all_cluster_labels = assign_clusters_to_all_examples(
        all_examples, 
        kmeans_model
    )
    
    # Step 6.5: Save all examples with cluster assignments and mismatch rates
    save_all_examples_with_clusters(
        all_examples,
        all_cluster_labels,
        cluster_stats,
        logged_indices
    )
    
    # Step 6.6: Plot mismatch rate distribution
    plot_mismatch_rate_distribution(cluster_stats)
    
    # Step 7: Filter high-mismatch clusters (threshold: 0.60 or max if none >= 0.60)
    filtered_examples = filter_high_mismatch_clusters(
        all_examples,
        all_cluster_labels,
        cluster_stats,
        logged_indices,
        threshold=0.60
    )
    
    # Step 8: Save filtered examples
    save_filtered_examples(filtered_examples, OUTPUT_PATH)
    
    # Step 9: Plot cluster distribution from predictions log
    if PREDICTIONS_LOG_PATH.exists():
        plot_prediction_cluster_distribution()
    else:
        print(f"\n⚠ Predictions log not found at {PREDICTIONS_LOG_PATH}")
        print(f"  Skipping prediction cluster distribution plot")
    
    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
