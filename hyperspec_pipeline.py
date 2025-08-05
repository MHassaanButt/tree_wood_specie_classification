#!/usr/bin/env python3
# hyperspec_pipeline.py

import os
import argparse
import logging
from glob import glob

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def setup_logger():
    logger = logging.getLogger("HyperspecPipeline")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def load_cubes(input_dir, logger):
    files = glob(os.path.join(input_dir, "*.npz"))
    if not files:
        logger.error(f"No .npz files found in {input_dir}")
        raise FileNotFoundError(f"No .npz files in {input_dir}")
    cubes = {}
    for fn in files:
        key = os.path.splitext(os.path.basename(fn))[0]
        data = np.load(fn)
        cube = data[list(data.keys())[0]]
        logger.debug(f"Loaded '{key}' shape={cube.shape}")
        cubes[key] = cube
    return cubes


def preprocess(cube, logger):
    h, w, bands = cube.shape
    logger.info(f"Cube has {bands} spectral bands")
    flat = cube.reshape(-1, bands)
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0) + 1e-8
    flat_norm = (flat - mu) / sigma
    return flat_norm, h, w


def apply_pca(flat, n_comp, logger):
    pca = PCA(n_components=n_comp)
    proj = pca.fit_transform(flat)
    var = pca.explained_variance_ratio_
    logger.info(f"PCA → {n_comp} comps explain {var.sum():.2%} of variance")
    return proj, var


def apply_kmeans(proj, k, logger):
    logger.info(f"Running KMeans with k={k}")
    km = KMeans(n_clusters=k, random_state=42)
    return km.fit_predict(proj)


def visualize_and_save(cube, proj, labels, h, w, key, out_dir, band_count):
    pca_img = proj.reshape(h, w, -1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # 1) First spectral band
    axes[0].imshow(cube[:, :, 0], cmap="gray")
    axes[0].set_title(f"{key} — Band 1")
    axes[0].axis("off")

    # 2) PCA RGB composite
    comps = min(3, pca_img.shape[-1])
    rgb = np.stack([pca_img[:, :, i] for i in range(comps)], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    axes[1].imshow(rgb)
    axes[1].set_title(f"{key} — PCA RGB ({band_count} bands)")
    axes[1].axis("off")

    # 3) KMeans clustering
    clust_img = labels.reshape(h, w)
    axes[2].imshow(clust_img, cmap="tab10")
    axes[2].set_title(f"{key} — KMeans ({band_count} bands)")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{key}_results.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised hyperspectral tree-ring clustering over multiple PCA band counts"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        default="prepocessing/processed_cubes",
        help="Relative path to .npz cubes (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="clustering_results",
        help="Parent folder for all band-specific results (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--bands",
        required=True,
        help="Comma-separated list of PCA component counts, e.g. 5,10,15",
    )
    parser.add_argument(
        "-k",
        "--n_clusters",
        type=int,
        default=2,
        help="Cluster count for KMeans (default: %(default)s)",
    )
    args = parser.parse_args()

    # parse band counts
    try:
        band_counts = [int(x) for x in args.bands.split(",") if x.strip()]
    except ValueError:
        raise ValueError("Bands must be comma-separated integers, e.g. 5,10,15")

    logger = setup_logger()
    cubes = load_cubes(args.input_dir, logger)

    # ensure top-level parent exists
    os.makedirs(args.output_dir, exist_ok=True)

    for b in band_counts:
        # now nested under clustering_results/
        out_dir = os.path.join(args.output_dir, f"bands_{b}")
        os.makedirs(out_dir, exist_ok=True)
        logger.info(
            f"=== Running pipeline for {b} bands → writing into '{out_dir}' ==="
        )

        for key, cube in cubes.items():
            logger.info(f"--- Processing '{key}' with {b} PCA bands ---")
            flat, h, w = preprocess(cube, logger)
            proj, var = apply_pca(flat, b, logger)
            labels = apply_kmeans(proj, args.n_clusters, logger)
            visualize_and_save(cube, proj, labels, h, w, key, out_dir, b)


if __name__ == "__main__":
    main()
