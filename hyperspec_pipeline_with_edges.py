#!/usr/bin/env python3
import os
import argparse
import logging

import numpy as np
import cv2
from glob import glob
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def setup_logger():
    log = logging.getLogger("HyperspecPipelineEdges")
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(ch)
    return log


def load_cubes(input_dir, logger):
    files = glob(os.path.join(input_dir, "*.npz"))
    if not files:
        logger.error(f"No .npz files in {input_dir}")
        raise FileNotFoundError(input_dir)
    cubes = {}
    for fn in files:
        key = os.path.splitext(os.path.basename(fn))[0]
        data = np.load(fn)
        cube = data[list(data.keys())[0]]
        logger.info(f"Loaded '{key}' shape={cube.shape}")
        cubes[key] = cube
    return cubes


def preprocess(cube, logger):
    H, W, B = cube.shape
    flat = cube.reshape(-1, B).astype(float)
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0) + 1e-8
    flat_norm = (flat - mu) / sigma
    logger.info(f"Preprocessed cube to shape {(H*W, B)}")
    return flat_norm, H, W


def apply_pca(flat, n_comp, logger):
    pca = PCA(n_components=n_comp)
    proj = pca.fit_transform(flat)
    logger.info(
        f"PCA: {n_comp} components, explained var {pca.explained_variance_ratio_.sum():.2%}"
    )
    return proj, pca


def apply_kmeans(proj, k, logger):
    logger.info(f"KMeans: k={k}")
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(proj)
    return labels


def compute_edges(pca_img_gray, low_thresh=50, high_thresh=150):
    """
    Run Canny edge detection on a normalized grayscale image.
    Inputs in 0–1 float range, outputs binary edge map.
    """
    # convert to 0–255 uint8
    uint8 = np.clip(pca_img_gray * 255, 0, 255).astype(np.uint8)
    edges = cv2.Canny(uint8, low_thresh, high_thresh)
    return edges


def visualize_and_save(cube, pca_proj, pca_model, labels, H, W, key, out_dir, n_comp):
    # rebuild PCA image
    img3 = pca_proj.reshape(H, W, n_comp)
    # PCA grayscale = first component
    gray = img3[:, :, 0]
    # normalize to 0–1
    gray_norm = (gray - gray.min()) / (gray.max() - gray.min())
    # edge map
    edges = compute_edges(gray_norm)

    # cluster map
    km_img = labels.reshape(H, W)

    # original band1 normalized for display
    band1 = cube[:, :, 0]
    b1n = (band1 - band1.min()) / (band1.max() - band1.min())

    # create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(b1n, cmap="gray")
    axes[0].set_title(f"{key} — Band 1")
    axes[0].axis("off")

    axes[1].imshow(gray_norm, cmap="gray")
    axes[1].set_title(f"{key} — PC1 Grayscale")
    axes[1].axis("off")

    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title(f"{key} — Canny Edges")
    axes[2].axis("off")

    # overlay edges in red onto band1
    overlay = np.dstack([b1n, b1n, b1n])
    overlay[edges > 0] = [1.0, 0, 0]
    axes[3].imshow(overlay)
    axes[3].set_title(f"{key} — Edges on Band1")
    axes[3].axis("off")

    plt.tight_layout()
    savepath = os.path.join(out_dir, f"{key}_edges.png")
    fig.savefig(savepath, dpi=150)
    plt.close(fig)

    # also save clustering + edges overlay
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 10))
    cmap = plt.cm.tab10
    ax2.imshow(km_img, cmap=cmap, alpha=0.6)
    ax2.imshow(overlay, alpha=0.4)
    ax2.set_title(f"{key} — KMeans & Edges ({n_comp} PCs)")
    ax2.axis("off")
    savepath2 = os.path.join(out_dir, f"{key}_cluster_edges.png")
    fig2.savefig(savepath2, dpi=150)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(
        description="HSI pipeline with PCA, KMeans & edge detection"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        default="prepocessing/processed_cubes",
        help="Directory containing .npz cubes",
    )
    parser.add_argument(
        "-o", "--output_dir", default="outputs_edges", help="Where to save results"
    )
    parser.add_argument(
        "-c", "--n_components", type=int, default=3, help="Number of PCA components"
    )
    parser.add_argument(
        "-k", "--n_clusters", type=int, default=2, help="Number of KMeans clusters"
    )
    parser.add_argument("--low", type=int, default=50, help="Canny low thresh")
    parser.add_argument("--high", type=int, default=150, help="Canny high thresh")
    args = parser.parse_args()

    logger = setup_logger()
    cubes = load_cubes(args.input_dir, logger)
    os.makedirs(args.output_dir, exist_ok=True)

    for key, cube in cubes.items():
        logger.info(f"--- Processing '{key}' ---")
        flat, H, W = preprocess(cube, logger)
        proj, pca_model = apply_pca(flat, args.n_components, logger)
        labels = apply_kmeans(proj, args.n_clusters, logger)
        visualize_and_save(
            cube, proj, pca_model, labels, H, W, key, args.output_dir, args.n_components
        )


if __name__ == "__main__":
    main()
