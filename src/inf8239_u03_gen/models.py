from __future__ import annotations


def build_autoencoder_parts(tf, latent_dim: int = 16):
    """Autoencoder denso con codificador y decodificador accesibles por separado."""
    inputs = tf.keras.Input((28, 28, 1))
    values = tf.keras.layers.Flatten()(inputs)
    latent = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent")(values)
    encoder = tf.keras.Model(inputs, latent, name=f"ae_encoder_{latent_dim}")
    latent_inputs = tf.keras.Input((latent_dim,))
    values = tf.keras.layers.Dense(784, activation="sigmoid")(latent_inputs)
    outputs = tf.keras.layers.Reshape((28, 28, 1))(values)
    decoder = tf.keras.Model(latent_inputs, outputs, name=f"ae_decoder_{latent_dim}")
    autoencoder = tf.keras.Model(inputs, decoder(encoder(inputs)), name=f"autoencoder_{latent_dim}")
    return encoder, decoder, autoencoder


def build_autoencoder(tf, latent_dim: int = 16):
    return build_autoencoder_parts(tf, latent_dim)[2]


def build_encoder_decoder(tf, latent_dim: int = 2):
    inputs = tf.keras.Input((28, 28, 1))
    values = tf.keras.layers.Flatten()(inputs)
    values = tf.keras.layers.Dense(128, activation="relu")(values)
    mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(values)
    log_variance = tf.keras.layers.Dense(latent_dim, name="z_log_var")(values)
    encoder = tf.keras.Model(inputs, [mean, log_variance], name=f"vae_encoder_{latent_dim}")
    latent_inputs = tf.keras.Input((latent_dim,))
    values = tf.keras.layers.Dense(128, activation="relu")(latent_inputs)
    values = tf.keras.layers.Dense(784, activation="sigmoid")(values)
    outputs = tf.keras.layers.Reshape((28, 28, 1))(values)
    decoder = tf.keras.Model(latent_inputs, outputs, name=f"vae_decoder_{latent_dim}")
    return encoder, decoder


def kl_divergence(tf, mean, log_variance):
    """KL entre N(mean, exp(log_variance)) y N(0, I), promediada sobre el lote."""
    return -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + log_variance - tf.square(mean) - tf.exp(log_variance), axis=1))


def make_vae(tf, encoder, decoder, beta: float = 1.0):
    """Construye el VAE: pérdida = reconstrucción + beta * KL, con reparametrización."""

    class VAE(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.beta = beta
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")
            self.reconstruction_tracker = tf.keras.metrics.Mean(name="reconstruction")
            self.kl_tracker = tf.keras.metrics.Mean(name="kl")

        @property
        def metrics(self):
            return [self.loss_tracker, self.reconstruction_tracker, self.kl_tracker]

        def train_step(self, data):
            images = data[0] if isinstance(data, tuple) else data
            with tf.GradientTape() as tape:
                mean, log_variance = self.encoder(images, training=True)
                epsilon = tf.random.normal(tf.shape(mean))
                latent = mean + tf.exp(0.5 * log_variance) * epsilon
                reconstruction = self.decoder(latent, training=True)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.keras.losses.binary_crossentropy(images, reconstruction), axis=(1, 2)))
                kl_loss = kl_divergence(tf, mean, log_variance)
                total_loss = reconstruction_loss + self.beta * kl_loss
            gradients = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
            self.loss_tracker.update_state(total_loss)
            self.reconstruction_tracker.update_state(reconstruction_loss)
            self.kl_tracker.update_state(kl_loss)
            return {metric.name: metric.result() for metric in self.metrics}

    return VAE()


def build_classifier(tf):
    """Clasificador pequeño usado solo como regla de medir: ¿las muestras parecen una clase reconocible?"""
    inputs = tf.keras.Input((28, 28, 1))
    values = tf.keras.layers.Flatten()(inputs)
    values = tf.keras.layers.Dense(128, activation="relu")(values)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(values)
    return tf.keras.Model(inputs, outputs, name="proxy_classifier")
