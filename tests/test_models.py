import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from inf8239_u03_gen.models import (
    build_autoencoder,
    build_autoencoder_parts,
    build_encoder_decoder,
    kl_divergence,
    make_vae,
)


def test_autoencoder_output_contract():
    model = build_autoencoder(tf)
    result = model(np.zeros((2, 28, 28, 1), dtype="float32")).numpy()
    assert result.shape == (2, 28, 28, 1)


def test_decoder_output_contract():
    _, decoder = build_encoder_decoder(tf, latent_dim=2)
    result = decoder(np.zeros((2, 2), dtype="float32")).numpy()
    assert result.shape == (2, 28, 28, 1)


def test_autoencoder_parts_are_consistent():
    encoder, decoder, autoencoder = build_autoencoder_parts(tf, latent_dim=4)
    batch = np.random.default_rng(0).random((3, 28, 28, 1)).astype("float32")
    latent = encoder(batch).numpy()
    assert latent.shape == (3, 4) and (latent >= 0).all()
    assert np.allclose(decoder(latent).numpy(), autoencoder(batch).numpy(), atol=1e-5)


def test_kl_is_zero_for_standard_normal_and_positive_otherwise():
    zeros = tf.zeros((4, 2))
    assert float(kl_divergence(tf, zeros, zeros)) == pytest.approx(0, abs=1e-6)
    assert float(kl_divergence(tf, tf.ones((4, 2)) * 2, zeros)) > 0


def test_vae_train_step_reports_components_and_changes_weights():
    encoder, decoder = build_encoder_decoder(tf, latent_dim=2)
    vae = make_vae(tf, encoder, decoder)
    vae.compile(optimizer="adam")
    batch = np.random.default_rng(1).random((32, 28, 28, 1)).astype("float32")
    before = [w.numpy().copy() for w in decoder.trainable_weights]
    history = vae.fit(batch, epochs=1, batch_size=16, verbose=0).history
    assert {"loss", "reconstruction", "kl"} <= set(history)
    assert history["loss"][0] == pytest.approx(history["reconstruction"][0] + history["kl"][0], rel=1e-3)
    assert any(not np.allclose(b, w.numpy()) for b, w in zip(before, decoder.trainable_weights))


def test_vae_is_reproducible_with_same_seed():
    def run():
        tf.keras.utils.set_random_seed(7)
        encoder, decoder = build_encoder_decoder(tf, latent_dim=2)
        vae = make_vae(tf, encoder, decoder)
        vae.compile(optimizer="adam")
        batch = np.random.default_rng(1).random((32, 28, 28, 1)).astype("float32")
        return vae.fit(batch, epochs=1, batch_size=16, shuffle=False, verbose=0).history["loss"][0]

    assert run() == pytest.approx(run(), rel=1e-4)
