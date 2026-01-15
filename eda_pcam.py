from pathlib import Path

import h5py
import numpy as np

# Gebruik matplotlib in "headless" mode (Snellius)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = Path.home() / "surfdrive"
OUT_DIR = Path.cwd() / "eda_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_X = DATA_DIR / "camelyonpatch_level_2_split_train_x.h5"
TRAIN_Y = DATA_DIR / "camelyonpatch_level_2_split_train_y.h5"


def _first_dataset(h5file):
    # pakt "x" of "y" als die er is, anders de eerste dataset in de file
    if "x" in h5file:
        return h5file["x"]
    if "y" in h5file:
        return h5file["y"]
    return list(h5file.values())[0]


def main():
    # --- Load a manageable subset for plots (sneller + genoeg voor EDA) ---
    n_show = 2000  # aantal samples voor histogram/label stats
    with h5py.File(TRAIN_X, "r") as fx, h5py.File(TRAIN_Y, "r") as fy:
        xds = _first_dataset(fx)
        yds = _first_dataset(fy)

        n_total = xds.shape[0]
        n = min(n_show, n_total)

        x = np.array(xds[:n])
        y = np.array(yds[:n]).reshape(-1).astype(int)

    print("=== BASIC INFO ===")
    print("Data dir:", DATA_DIR)
    print("Train X:", TRAIN_X)
    print("Train Y:", TRAIN_Y)
    print("Subset size used:", n)
    print("Images shape:", x.shape, "dtype:", x.dtype)
    print("Labels shape:", y.shape, "unique:", np.unique(y))
    print("Pixel min/max/mean:", x.min(), x.max(), float(x.mean()))

    # --- 1) Label distribution (counts + percentages) ---
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    print("\n=== LABEL DISTRIBUTION (subset) ===")
    for c, cnt in zip(classes, counts):
        print(f"class {c}: {cnt} ({cnt/total*100:.1f}%)")

    plt.figure()
    plt.bar(classes.astype(str), counts)
    plt.xlabel("Class label")
    plt.ylabel("Count")
    plt.title("PCAM label distribution (subset)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_label_distribution.png", dpi=150)
    plt.close()

    # --- 2) Pixel value histogram (raw) ---
    plt.figure()
    plt.hist(x.reshape(-1), bins=50)
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.title("Pixel value distribution (subset, raw)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_pixel_histogram_raw.png", dpi=150)
    plt.close()

    # --- 3) Example patches (balanced view: a few per class) ---
    # Kies 6 voorbeelden: 3 neg + 3 pos (als aanwezig)
    idx0 = np.where(y == 0)[0][:3]
    idx1 = np.where(y == 1)[0][:3]
    idxs = np.concatenate([idx0, idx1]) if len(idx1) > 0 else idx0

    plt.figure(figsize=(8, 4))
    for i, idx in enumerate(idxs, start=1):
        plt.subplot(2, 3, i)
        plt.imshow(x[idx])
        plt.title(f"y={y[idx]}")
        plt.axis("off")
    plt.suptitle("Example PCAM patches (subset)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_example_patches.png", dpi=150)
    plt.close()

    # --- 4) (Optional) Mean intensity per class (quick artifact check) ---
    # (heel handig om “te zwart/te wit” outliers te zien)
    means = x.mean(axis=(1, 2, 3))
    plt.figure()
    plt.hist(means[y == 0], bins=40, alpha=0.7, label="class 0")
    if (y == 1).any():
        plt.hist(means[y == 1], bins=40, alpha=0.7, label="class 1")
    plt.xlabel("Mean pixel intensity")
    plt.ylabel("Count")
    plt.title("Mean intensity distribution by class (subset)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_mean_intensity_by_class.png", dpi=150)
    plt.close()

    print(f"\nSaved plots to: {OUT_DIR.resolve()}")
    print("Files:")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
