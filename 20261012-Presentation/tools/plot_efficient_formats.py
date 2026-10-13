"""Generate assets/img/efficient-formats.svg with NumPy, SciPy and Matplotlib.

Fits are cached in assets/data/efficient-formats.json for fast standalone redraws.
Bump VERSION after changing the fitting algorithm to invalidate the cache.
Fits use 200,000 independent Student-t(7) pairs, seed 100, population RMS 1.
Scalar: 8 symmetric Lloyd-Max centroids with a fitted global scale.
Block: 4 symmetric element centroids and 4 shared scales fitted jointly.
VQ: 64 centroids fitted by k-means++, best of four Lloyd runs.
All formats use 3 bits/value, including the block's 2 scale bits per pair.
Global scales and codebook storage overhead are excluded from that budget.
RMSE is recomputed on the same independent, untrimmed held-out sample.
"""

import json
from pathlib import Path
from time import perf_counter

START = perf_counter()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from scipy.spatial import cKDTree
from scipy.stats import t


ROOT = Path(__file__).resolve().parent.parent
SEED = 100
DF = 7
N_TRAIN = 200_000
N_TEST = 200_000

VERSION = 1
CACHE = ROOT / "assets" / "data" / "efficient-formats.json"


def main():
    rng = np.random.default_rng(SEED)
    unit_rms = np.sqrt((DF - 2) / DF)
    train = rng.standard_t(DF, size=(N_TRAIN, 2)) * unit_rms
    test = rng.standard_t(DF, size=(N_TEST, 2)) * unit_rms
    fit_config = {"seed": SEED, "df": DF, "n_train": N_TRAIN}
    cached = json.loads(CACHE.read_text()) if CACHE.exists() else None
    if cached is not None and cached["version"] == VERSION and cached["fit_config"] == fit_config:
        print(f"Using cached fits: {CACHE}")
    else:
        print("Fitting codebooks (cache missing or outdated)...", flush=True)
        # Symmetric scalar Lloyd-Max: fit four positive magnitudes, then reflect.
        magnitudes = np.abs(train.ravel())
        centroids = np.quantile(magnitudes, [0.15, 0.45, 0.75, 0.95])
        for _ in range(500):
            indices = np.searchsorted((centroids[1:] + centroids[:-1]) / 2, magnitudes)
            updated = np.bincount(indices, weights=magnitudes, minlength=4)
            updated /= np.bincount(indices, minlength=4)
            change = np.max(np.abs(updated - centroids))
            centroids = updated
            if change < 1e-7:
                break
        centroids = np.concatenate((-centroids[::-1], centroids)) / centroids[-1]
        tree = cKDTree(centroids[:, None])

        def scalar_mse(log_scale):
            scale = np.exp(log_scale)
            distance, _ = tree.query(train.ravel()[:, None] / scale)
            return np.mean((scale * distance) ** 2)

        candidates = np.linspace(np.log(0.25), np.log(32), 65)
        best = np.argmin([scalar_mse(value) for value in candidates])
        fit = minimize_scalar(
            scalar_mse, method="bounded",
            bounds=(candidates[max(0, best - 1)], candidates[min(64, best + 1)]),
            options={"xatol": 1e-7},
        )
        scalar_centroids = centroids * np.exp(fit.x)
        print("Scalar fit complete", flush=True)

        # Fix the outer element centroid at 1 to remove scale degeneracy.
        # Optimise the inner magnitude and four increasing shared scales jointly.
        def block_codebook(parameters):
            inner = 1 / (1 + np.exp(-parameters[0]))
            scales = np.cumsum(np.exp(parameters[1:]))
            element_centroids = np.array([-1, -inner, inner, 1])
            pairs = np.stack(np.meshgrid(element_centroids, element_centroids), axis=-1).reshape(-1, 2)
            codebook = (scales[:, None, None] * pairs[None, :, :]).reshape(-1, 2)
            return element_centroids, scales, codebook

        def block_mse(parameters, data):
            distance, _ = cKDTree(block_codebook(parameters)[2]).query(data)
            return np.mean(distance ** 2) / 2

        solutions = []
        for seed in [102, 103, 104]:
            initial = differential_evolution(
                block_mse, bounds=[(-3, 2)] * 5, args=(train[:30_000],),
                rng=np.random.default_rng(seed), popsize=10, maxiter=100, tol=2e-6,
            )
            fit = minimize(
                block_mse, initial.x, args=(train,), method="Nelder-Mead",
                options={"xatol": 2e-6, "fatol": 1e-11, "maxiter": 800},
            )
            solutions.append(fit)
        fit = min(solutions, key=lambda solution: solution.fun)
        block_element_centroids, block_scales, _ = block_codebook(fit.x)
        print("Joint block fit complete", flush=True)

        # k-means++ and Lloyd updates, retaining the best of four starts.
        best_error = np.inf
        for _ in range(4):
            centres = np.empty((64, 2))
            centres[0] = train[rng.integers(N_TRAIN)]
            squared_distance = np.full(N_TRAIN, np.inf)
            for index in range(1, 64):
                squared_distance = np.minimum(
                    squared_distance, np.sum((train - centres[index - 1]) ** 2, axis=1)
                )
                centres[index] = train[rng.choice(N_TRAIN, p=squared_distance / squared_distance.sum())]
            for _ in range(300):
                distance, indices = cKDTree(centres).query(train)
                counts = np.bincount(indices, minlength=64)
                updated = np.column_stack([
                    np.bincount(indices, weights=train[:, axis], minlength=64)
                    for axis in range(2)
                ])
                occupied = counts > 0
                updated[occupied] /= counts[occupied, None]
                if not occupied.all():
                    updated[~occupied] = train[np.argsort(distance)[-np.sum(~occupied):]]
                change = np.max(np.linalg.norm(updated - centres, axis=1))
                centres = updated
                if change < 1e-5:
                    break
            error = np.mean(cKDTree(centres).query(train)[0] ** 2)
            if error < best_error:
                best_error = error
                vector_centroids = centres.copy()
        cached = {
            "version": VERSION,
            "fit_config": fit_config,
            "scalar_centroids": scalar_centroids.tolist(),
            "block_element_centroids": block_element_centroids.tolist(),
            "block_scales": block_scales.tolist(),
            "vector_centroids": vector_centroids.tolist(),
        }
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cached, indent=2) + "\n")
        print(f"Saved fits: {CACHE}", flush=True)

    scalar_centroids = np.array(cached["scalar_centroids"])
    block_element_centroids = np.array(cached["block_element_centroids"])
    block_scales = np.array(cached["block_scales"])
    vector_centroids = np.array(cached["vector_centroids"])

    scalar_pairs = np.stack(np.meshgrid(scalar_centroids, scalar_centroids), axis=-1).reshape(-1, 2)
    block_pairs = np.stack(np.meshgrid(block_element_centroids, block_element_centroids), axis=-1).reshape(-1, 2)
    block = (block_scales[:, None, None] * block_pairs[None, :, :]).reshape(-1, 2)
    codebooks = [scalar_pairs, block, vector_centroids]
    rmses = [np.sqrt(np.mean(cKDTree(codebook).query(test)[0] ** 2) / 2) for codebook in codebooks]

    # Identical bounds include every centroid, including the learned tail values.
    limit = np.ceil(max(np.abs(codebook).max() for codebook in codebooks) * 1.05 * 2) / 2
    axis = np.linspace(-limit, limit, 600)
    marginal = t.pdf(axis / unit_rms, DF) / unit_rms
    density = marginal[:, None] * marginal[None, :]
    cmap = LinearSegmentedColormap.from_list("slide_blue_grey", ["#ffffff", "#dbe5ec", "#a3bbcb", "#64899f"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "text.color": "#252530",
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.12, wspace=0.10)
    titles = ["Nonlinear scalar", "2D Block scaled", "2D Vector quantisation"]
    for ax, title, codebook, rmse in zip(axes, titles, codebooks, rmses):
        ax.imshow(
            density, origin="lower", extent=(-limit, limit, -limit, limit),
            cmap=cmap, norm=PowerNorm(gamma=0.22), interpolation="bilinear",
        )
        ax.scatter(*codebook.T, s=18, facecolor="#252530", edgecolor="white", linewidth=0.45)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#85858f")
            spine.set_linewidth(0.6)
        ax.text(0.5, 1.055, title, transform=ax.transAxes, ha="center", fontsize=17)
        ax.text(0.5, -0.10, f"RMSE {rmse:.3f}", transform=ax.transAxes, ha="center", fontsize=16)

    output = ROOT / "assets" / "img" / "efficient-formats.svg"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    print(output)
    for title, codebook, rmse in zip(titles, codebooks, rmses):
        print(f"{title}: {len(codebook)} centroids, held-out per-value RMSE {rmse:.6f}")
    print(f"Elapsed: {perf_counter() - START:.2f} s")


if __name__ == "__main__":
    main()
