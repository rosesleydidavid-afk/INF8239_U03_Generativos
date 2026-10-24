"""LAB10: autoencoder y VAE con Fashion-MNIST, con evaluación numérica, línea base PCA, diversidad y memoria."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from inf8239_u03_gen.data import normalize_images, validate_images
from inf8239_u03_gen.metrics import (
    class_entropy,
    fixed_indices,
    fraction_below,
    mean_pairwise_distance,
    mse,
    nearest_neighbor,
    pair_with_different_labels,
    pca_reconstruction,
    per_class_mse,
    psnr,
)
from inf8239_u03_gen.models import (
    build_autoencoder_parts,
    build_classifier,
    build_encoder_decoder,
    make_vae,
)

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
BATCH = 256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-ae", type=int, default=6)
    parser.add_argument("--epochs-vae", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="usa solo N imágenes de entrenamiento")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latents", type=int, nargs="+", default=[2, 16],
                        help="tamaños latentes a comparar (debe incluir 2 para la aplicación)")
    parser.add_argument("--samples", type=int, default=1000, help="muestras generadas por modelo")
    parser.add_argument("--panel-size", type=int, default=16)
    parser.add_argument("--quick", action="store_true", help="prueba rápida: 3000 imágenes, 1 época, 200 muestras")
    args = parser.parse_args()
    if args.quick:
        args.limit, args.epochs_ae, args.epochs_vae, args.samples = 3000, 1, 1, 200
    if 2 not in args.latents:
        parser.error("--latents debe incluir 2 (la aplicación usa un decodificador de 2 dimensiones)")
    return args


tf.get_logger().setLevel("ERROR")


def seed_everything(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def history_to_dict(history) -> dict:
    return {key: [float(v) for v in values] for key, values in history.history.items()}


def fit_autoencoder(latent, x_train, epochs, seed):
    seed_everything(seed)
    encoder, decoder, model = build_autoencoder_parts(tf, latent)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    start = perf_counter()
    history = model.fit(x_train, x_train, validation_split=0.1, epochs=epochs, batch_size=BATCH,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
                        verbose=2)
    return encoder, decoder, model, perf_counter() - start, history_to_dict(history)


def fit_vae(latent, x_train, epochs, seed):
    seed_everything(seed)
    encoder, decoder = build_encoder_decoder(tf, latent_dim=latent)
    vae = make_vae(tf, encoder, decoder)
    vae.compile(optimizer="adam")
    start = perf_counter()
    history = vae.fit(x_train, epochs=epochs, batch_size=BATCH, verbose=2)
    return encoder, decoder, vae, perf_counter() - start, history_to_dict(history)


def decode_ms_per_image(decoder, latent_vectors, repeats=5) -> float:
    decoder.predict(latent_vectors[:64], verbose=0)  # calentamiento
    times = []
    for _ in range(repeats):
        start = perf_counter()
        decoder.predict(latent_vectors, verbose=0, batch_size=1024)
        times.append(perf_counter() - start)
    return 1000 * median(times) / len(latent_vectors)


def sample_metrics(samples, x_train, classifier, real_reference):
    distances, indices = nearest_neighbor(samples, x_train)
    probabilities = classifier.predict(samples, verbose=0, batch_size=1024)
    predicted = probabilities.argmax(axis=1)
    counts = np.bincount(predicted, minlength=10)
    diversity = mean_pairwise_distance(samples)
    return {
        "nn_distance_mean": float(distances.mean()),
        "nn_distance_median": float(np.median(distances)),
        "share_closer_than_real_p05": fraction_below(distances, real_reference["nn_distance_p05"]),
        "diversity_pairwise": diversity,
        "diversity_ratio_vs_real": diversity / real_reference["diversity_pairwise"],
        "class_entropy_nats": class_entropy(counts),
        "classes_present": int((counts > 0).sum()),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "class_counts": counts.tolist(),
    }, distances, indices


def grid_figure(path, rows, title=None, columns=None, size=0.9):
    n_rows = len(rows)
    columns = columns or max(len(images) for _, images in rows)
    figure, axes = plt.subplots(n_rows, columns, figsize=(columns * size + 1.6, n_rows * size + 0.5), squeeze=False)
    for r, (label, images) in enumerate(rows):
        for c in range(columns):
            axis = axes[r, c]
            axis.axis("off")
            if c < len(images):
                axis.imshow(np.squeeze(images[c]), cmap="gray", vmin=0, vmax=1)
        axes[r, 0].text(-0.15, 0.5, label, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
    if title:
        figure.suptitle(title, fontsize=10)
    figure.subplots_adjust(left=0.16, right=0.99, top=0.9 if title else 0.98, bottom=0.02, wspace=0.05, hspace=0.05)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main():
    args = parse_args()
    reports = ROOT / "reports"
    models = ROOT / "models"
    reports.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    seed_everything(args.seed)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = normalize_images(x_train), normalize_images(x_test)
    if args.limit:
        x_train, y_train = x_train[: args.limit], y_train[: args.limit]
        x_test, y_test = x_test[: max(100, args.limit // 5)], y_test[: max(100, args.limit // 5)]
    validate_images(x_train)
    validate_images(x_test)
    n_samples = min(args.samples, len(x_test))
    panel_idx = fixed_indices(len(x_test), args.panel_size, args.seed)
    pair_a, pair_b = pair_with_different_labels(y_test, args.seed)
    print(f"Entrenamiento: {len(x_train)} · Prueba: {len(x_test)} · Muestras por modelo: {n_samples}")

    # Clasificador de referencia: solo sirve para medir si las muestras parecen una clase reconocible.
    seed_everything(args.seed)
    classifier = build_classifier(tf)
    classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    classifier.fit(x_train, y_train, epochs=3, batch_size=BATCH, verbose=0)
    classifier_accuracy = float(classifier.evaluate(x_test, y_test, verbose=0)[1])

    # Referencia con imágenes reales: ¿qué tan lejos están de su vecino de entrenamiento imágenes reales nuevas?
    real_idx = fixed_indices(len(x_test), n_samples, args.seed + 1)
    real_images = x_test[real_idx]
    real_distances, _ = nearest_neighbor(real_images, x_train)
    real_probabilities = classifier.predict(real_images, verbose=0, batch_size=1024)
    real_counts = np.bincount(real_probabilities.argmax(axis=1), minlength=10)
    real_reference = {
        "nn_distance_mean": float(real_distances.mean()),
        "nn_distance_median": float(np.median(real_distances)),
        "nn_distance_p05": float(np.percentile(real_distances, 5)),
        "diversity_pairwise": mean_pairwise_distance(real_images),
        "class_entropy_nats": class_entropy(real_counts),
        "mean_confidence": float(real_probabilities.max(axis=1).mean()),
    }

    results, histories, samples_by_model, nn_by_model = {}, {}, {}, {}
    reconstructions = {}
    encoders = {}

    for latent in sorted(args.latents):
        # --- Autoencoder ---
        encoder, decoder, model, seconds, history = fit_autoencoder(latent, x_train, args.epochs_ae, args.seed)
        name = f"AE-{latent}"
        rebuilt = model.predict(x_test, verbose=0, batch_size=1024)
        z = np.random.default_rng(args.seed).normal(size=(n_samples, latent)).astype("float32")
        samples = decoder.predict(z, verbose=0, batch_size=1024)
        sm, distances, indices = sample_metrics(samples, x_train, classifier, real_reference)
        results[name] = {
            "kind": "autoencoder", "latent_dim": latent, "train_seconds": seconds,
            "parameters": int(model.count_params()), "recon_mse": mse(x_test, rebuilt),
            "recon_psnr_db": psnr(x_test, rebuilt), "recon_mse_per_class": per_class_mse(x_test, rebuilt, y_test),
            "decode_ms_per_image": decode_ms_per_image(decoder, z), "sampling": sm,
            "note": "El AE no impone una distribución latente: muestrear N(0, I) no es su uso previsto.",
        }
        histories[name] = history
        reconstructions[name] = rebuilt[panel_idx]
        samples_by_model[name], nn_by_model[name] = samples, (distances, indices)
        encoders[name] = (encoder, decoder, model)

        # --- VAE ---
        encoder, decoder, vae, seconds, history = fit_vae(latent, x_train, args.epochs_vae, args.seed)
        name = f"VAE-{latent}"
        mean_test, _ = encoder.predict(x_test, verbose=0, batch_size=1024)
        rebuilt = decoder.predict(mean_test, verbose=0, batch_size=1024)
        samples = decoder.predict(z, verbose=0, batch_size=1024)
        sm, distances, indices = sample_metrics(samples, x_train, classifier, real_reference)
        results[name] = {
            "kind": "vae", "latent_dim": latent, "train_seconds": seconds,
            "parameters": int(encoder.count_params() + decoder.count_params()), "recon_mse": mse(x_test, rebuilt),
            "recon_psnr_db": psnr(x_test, rebuilt), "recon_mse_per_class": per_class_mse(x_test, rebuilt, y_test),
            "decode_ms_per_image": decode_ms_per_image(decoder, z), "sampling": sm,
            "final_reconstruction_loss": history["reconstruction"][-1], "final_kl": history["kl"][-1],
        }
        histories[name] = history
        reconstructions[name] = rebuilt[panel_idx]
        samples_by_model[name], nn_by_model[name] = samples, (distances, indices)
        encoders[name] = (encoder, decoder, vae)

        # --- Línea base lineal ---
        start = perf_counter()
        rebuilt = pca_reconstruction(x_train, x_test, latent, args.seed)
        name = f"PCA-{latent}"
        results[name] = {
            "kind": "pca", "latent_dim": latent, "train_seconds": perf_counter() - start,
            "parameters": int(latent * 784 + 784), "recon_mse": mse(x_test, rebuilt),
            "recon_psnr_db": psnr(x_test, rebuilt), "recon_mse_per_class": per_class_mse(x_test, rebuilt, y_test),
            "sampling": None, "note": "Línea base lineal; no genera muestras.",
        }
        reconstructions[name] = rebuilt[panel_idx]
    for entry in results.values():
        entry["parameter_memory_kb"] = entry["parameters"] * 4 / 1024

    # --- Paneles con ejemplos elegidos por semilla fija ---
    big = max(args.latents)
    originals = x_test[panel_idx]
    rows = [("Original", originals)]
    for name in (f"AE-{big}", "AE-2", f"VAE-{big}", "VAE-2", f"PCA-{big}", "PCA-2"):
        if name in reconstructions:
            rows.append((name, reconstructions[name]))
    grid_figure(reports / "panel_reconstruction.png", rows,
                title=f"Reconstrucción de {args.panel_size} imágenes de prueba (semilla {args.seed}, no seleccionadas a mano)")

    sample_rows = [(f"{n}", samples_by_model[n][: args.panel_size]) for n in (f"AE-{big}", "VAE-2", f"VAE-{big}")
                   if n in samples_by_model]
    grid_figure(reports / "panel_samples.png", sample_rows, title="Muestras decodificadas desde z ~ N(0, I)")

    # --- Interpolación entre dos imágenes reales de clases distintas ---
    steps = np.linspace(0, 1, 10)
    interpolation_rows = []
    for name in (f"AE-{big}", "VAE-2", f"VAE-{big}"):
        if name not in encoders:
            continue
        encoder, decoder, _ = encoders[name]
        pair = x_test[[pair_a, pair_b]]
        if name.startswith("VAE"):
            z_pair = encoder.predict(pair, verbose=0)[0]
        else:
            z_pair = encoder.predict(pair, verbose=0)
        path = np.array([(1 - t) * z_pair[0] + t * z_pair[1] for t in steps], dtype="float32")
        frames = decoder.predict(path, verbose=0)
        interpolation_rows.append((name, [x_test[pair_a], *frames, x_test[pair_b]]))
    grid_figure(reports / "interpolation.png", interpolation_rows, columns=12,
                title=f"Interpolación: {CLASS_NAMES[int(y_test[pair_a])]} → {CLASS_NAMES[int(y_test[pair_b])]}")

    # --- Vecino más cercano del entrenamiento para muestras del VAE ---
    nn_rows = []
    for name in ("VAE-2", f"VAE-{big}"):
        if name in nn_by_model:
            distances, indices = nn_by_model[name]
            nn_rows.append((f"{name}\nmuestra", samples_by_model[name][:12]))
            nn_rows.append(("vecino en\nentrenamiento", x_train[indices[:12]]))
    grid_figure(reports / "nearest_neighbors.png", nn_rows, columns=12,
                title="¿Copia o variación? Muestra y su vecino más cercano del entrenamiento")

    # --- Espacio latente de 2 dimensiones ---
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    subset = fixed_indices(len(x_test), min(5000, len(x_test)), args.seed + 2)
    for axis, name in zip(axes, ("AE-2", "VAE-2")):
        encoder = encoders[name][0]
        z_test = encoder.predict(x_test[subset], verbose=0)
        z_test = z_test[0] if name.startswith("VAE") else z_test
        axis.scatter(z_test[:, 0], z_test[:, 1], c=y_test[subset], cmap="tab10", vmin=0, vmax=9, s=4)
        axis.set_title(f"{name}: codificación de {len(subset)} imágenes de prueba")
        axis.set_xlabel("z1")
        axis.set_ylabel("z2")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=plt.cm.tab10(i / 9)) for i in range(10)]
    figure.legend(handles, CLASS_NAMES, loc="center right", fontsize=7, title="Clase")
    figure.tight_layout(rect=(0, 0, 0.88, 1))
    figure.savefig(reports / "latent_space_2d.png", dpi=170)
    plt.close(figure)

    # --- Curvas de entrenamiento ---
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for latent in sorted(args.latents):
        ae, vae = histories[f"AE-{latent}"], histories[f"VAE-{latent}"]
        axes[0].plot(ae["val_loss"], marker="o", label=f"AE-{latent} (validación)")
        axes[1].plot(vae["reconstruction"], marker="o", label=f"VAE-{latent}")
        axes[2].plot(vae["kl"], marker="o", label=f"VAE-{latent}")
    axes[0].set_title("AE: pérdida de validación (BCE por píxel)")
    axes[1].set_title("VAE: reconstrucción (BCE por imagen)")
    axes[2].set_title("VAE: divergencia KL")
    for axis in axes:
        axis.set_xlabel("Época")
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(reports / "training_curves.png", dpi=170)
    plt.close(figure)

    # --- Guardado de modelos y resultados ---
    encoders[f"AE-{big}"][2].save(models / "autoencoder.keras")
    encoders["VAE-2"][0].save(models / "encoder.keras")
    encoders["VAE-2"][1].save(models / "decoder.keras")
    for latent in args.latents:
        if latent != 2:
            encoders[f"VAE-{latent}"][1].save(models / f"decoder_vae{latent}.keras")

    run = {
        "arguments": vars(args),
        "environment": {"python": platform.python_version(), "os": platform.platform(),
                        "processor": platform.processor(), "tensorflow": tf.__version__,
                        "numpy": np.__version__, "gpu": [d.name for d in tf.config.list_physical_devices("GPU")]},
        "data": {"train_images": len(x_train), "test_images": len(x_test), "license": "Fashion-MNIST, MIT",
                 "panel_indices": panel_idx.tolist(),
                 "interpolation_pair": {"a": pair_a, "b": pair_b,
                                        "classes": [CLASS_NAMES[int(y_test[pair_a])], CLASS_NAMES[int(y_test[pair_b])]]}},
        "classifier_reference": {"test_accuracy": classifier_accuracy, "note": "regla de medir para muestras, no modelo evaluado"},
        "real_images_reference": real_reference,
        "models": results,
        "histories": histories,
    }
    (reports / "lab10_metrics.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

    columns = ["model", "kind", "latent_dim", "recon_mse", "recon_psnr_db", "train_seconds", "parameters",
               "parameter_memory_kb", "decode_ms_per_image", "nn_distance_mean", "share_closer_than_real_p05",
               "diversity_ratio_vs_real", "class_entropy_nats", "classes_present", "mean_confidence"]
    with open(reports / "lab10_summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for name, entry in results.items():
            row = {"model": name, **{k: entry.get(k) for k in columns if k in entry}}
            if entry["sampling"]:
                row.update({k: entry["sampling"][k] for k in columns if k in entry["sampling"]})
            writer.writerow(row)

    print("\n=== Reconstrucción en prueba (menor MSE es mejor) ===")
    for name, entry in results.items():
        print(f"{name:7s} MSE={entry['recon_mse']:.5f}  PSNR={entry['recon_psnr_db']:.2f} dB  "
              f"entrenamiento={entry['train_seconds']:.1f}s  parámetros={entry['parameters']}")
    print("\n=== Muestras desde N(0, I) frente a imágenes reales ===")
    print(f"Referencia real: distancia al vecino (mediana)={real_reference['nn_distance_median']:.3f} "
          f"· p05={real_reference['nn_distance_p05']:.3f} · entropía de clases={real_reference['class_entropy_nats']:.3f}")
    for name, entry in results.items():
        sm = entry["sampling"]
        if sm:
            print(f"{name:7s} dist. vecino (mediana)={sm['nn_distance_median']:.3f}  "
                  f"muy cercanas={sm['share_closer_than_real_p05']:.1%}  "
                  f"diversidad/real={sm['diversity_ratio_vs_real']:.2f}  "
                  f"clases={sm['classes_present']}/10  entropía={sm['class_entropy_nats']:.3f}  "
                  f"confianza={sm['mean_confidence']:.2f}")
    print(f"\nClasificador de referencia: exactitud en prueba = {classifier_accuracy:.3f}")
    print("Archivos en reports/: lab10_metrics.json, lab10_summary.csv, panel_reconstruction.png, panel_samples.png, "
          "interpolation.png, nearest_neighbors.png, latent_space_2d.png, training_curves.png")


if __name__ == "__main__":
    main()
