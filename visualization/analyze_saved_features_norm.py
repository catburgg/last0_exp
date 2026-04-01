import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from numpy.linalg import norm

def main():
    data_path = "/mnt/wfm/code/zxh/last0_exp/eval_latent_sim_out/latent_features_img_100.npy"
    data = np.load(data_path)
    N, T, D = data.shape

    seq_mean = np.mean(data, axis=1, keepdims=True) 
    data_centered = data - seq_mean                 

    data_norms = norm(data_centered, axis=-1, keepdims=True)
    data_normalized = data_centered / (data_norms + 1e-8)

    def cos_sim(a, b):
        return np.sum(a * b, axis=-1) / (norm(a, axis=-1) * norm(b, axis=-1) + 1e-8)

    print("\n" + "="*60)
    print("Intra-Sequence Temporal Cosine Similarity")
    print("="*60)
    for k in range(1, T):
        sims = cos_sim(data_normalized[:, 0, :], data_normalized[:, k, :])
        print(f"Step t+{k} : {np.mean(sims):.4f} ± {np.std(sims):.4f}")

    flat_normalized = data_normalized.reshape(N * T, D)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(flat_normalized)
    pca_reshaped = pca_result.reshape(N, T, 2)
    
    print("\n" + "="*60)
    print("PCA Inter-Cluster Distance Analysis (Normalized Space)")
    print("="*60)
    print(f"Explained Variance Ratio: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")
    
    centroids = np.mean(pca_reshaped, axis=0) # [T, 2]

    print("\n[Centroid-to-Centroid Distance from Current State (t)]")
    for k in range(1, T):
        dist = norm(centroids[k] - centroids[0])
        print(f"Distance (t vs t+{k}): {dist:.4f}")

    print("\n[Mean Point-to-Centroid Distance from Current State (t)]")
    for k in range(1, T):
        point_to_centroid_dists = norm(pca_reshaped[:, k, :] - centroids[0], axis=1)
        print(f"Mean Distance (t+{k} points vs t centroid): {np.mean(point_to_centroid_dists):.4f} ± {np.std(point_to_centroid_dists):.4f}")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7), dpi=300)
    colors = sns.color_palette("coolwarm", T)
    labels = ['Current (t)'] + [f'Future (t+{k})' for k in range(1, T)]
    
    plot_samples = min(N, 80) 
    for i in range(plot_samples):
        if norm(pca_reshaped[i, 0, :] - pca_reshaped[i, -1, :]) < 1e-4: continue
        plt.plot(pca_reshaped[i, :, 0], pca_reshaped[i, :, 1], color='gray', alpha=0.1, linewidth=1.0, zorder=1)
        for t_step in range(T):
            plt.scatter(pca_reshaped[i, t_step, 0], pca_reshaped[i, t_step, 1], 
                        color=colors[t_step], s=30, alpha=0.3, edgecolor='none', zorder=2)

    plt.plot(centroids[:, 0], centroids[:, 1], color='black', alpha=0.7, linewidth=3.0, linestyle='-', zorder=3, label="Macro Temporal Trend")
    
    for t_step in range(T):
        plt.scatter(centroids[t_step, 0], centroids[t_step, 1], 
                    color=colors[t_step], s=400, marker='*', edgecolor='black', linewidth=1.5, 
                    zorder=4, label=f"{labels[t_step]} Centroid")

    plt.title('Normalized PCA of Temporal Trajectories (Sequence-Aligned & L2 Scaled)', fontsize=15, pad=15)
    plt.xlabel(f'Temporal Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=13)
    plt.ylabel(f'Temporal Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=13)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=11)
    plt.tight_layout()
    
    save_path = os.path.join(
        os.path.dirname(data_path),
        "pca_temporal_trajectory_with_centroids_1_normalized.png",
    )
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nSaved visualization with cluster centroids to {save_path}")

if __name__ == "__main__":
    main()