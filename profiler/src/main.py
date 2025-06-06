import tsfel
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Step 1: Generate or load time-series data
def generate_synthetic_time_series():
    """Simulate synthetic time-series data for this example."""
    mean = 50
    noise_scale = 10
    N = 96
    noise = np.random.normal(0, noise_scale, N)
    return mean + noise


# Create synthetic time-series data
time_series_data = generate_synthetic_time_series()


# Step 2: Extract features from time-series using TSFEL
def extract_features(data):
    """Extract relevant features from time-series data using TSFEL."""
    df_data = pd.DataFrame(data, columns=["signal"])

    # Load TSFEL feature configuration
    feature_extraction_settings = tsfel.get_features_by_domain()

    # Extract features
    features = tsfel.time_series_features_extractor(
        feature_extraction_settings, df_data, verbose=False)
    return features


# Extract features
features = extract_features(time_series_data)

# Normalize features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features.values)

# Step 3: Build GAN model


def build_gan(input_dim):
    """Build GAN model."""
    # Generator
    generator = models.Sequential([
        layers.Dense(128, activation="relu", input_dim=input_dim),
        layers.Dense(256, activation="relu"),
        # Output shape matches the input_dim
        layers.Dense(input_dim, activation="tanh")
    ])

    # Discriminator
    discriminator = models.Sequential([
        layers.Dense(256, activation="relu", input_dim=input_dim),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # Binary classification output
    ])

    discriminator.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    return generator, discriminator


# Step 4: Train GAN
def train_gan(generator, discriminator, data, epochs=1000, batch_size=32):
    """Train the GAN to create synthetic data."""
    # Get input dimension
    input_dim = data.shape[1]

    # Combine generator and discriminator to form GAN
    discriminator.trainable = False
    gan = models.Sequential([generator, discriminator])
    gan.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001), loss="binary_crossentropy")

    # Training loop
    for epoch in range(epochs):
        # ---------------------------
        # Train Discriminator
        # ---------------------------
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_samples = data[idx]

        noise = np.random.normal(0, 1, (batch_size, input_dim))
        fake_samples = generator.predict(noise)

        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_samples, labels_real)
        d_loss_fake = discriminator.train_on_batch(fake_samples, labels_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------------
        # Train Generator (via GAN)
        # ---------------------------
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        labels_gan = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, labels_gan)

        # Print progress
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {
                  epoch}/{epochs}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")

    return generator


# Build and train GAN
input_dim = scaled_features.shape[1]
generator, discriminator = build_gan(input_dim)
trained_generator = train_gan(generator, discriminator, scaled_features)

# Step 5: Generate synthetic data


def generate_synthetic_data(generator, n_samples=100):
    """Generate synthetic data using the trained generator."""
    noise = np.random.normal(0, 1, (n_samples, generator.input_shape[1]))
    synthetic_data = generator.predict(noise)
    return synthetic_data


# Generate synthetic features
synthetic_features = generate_synthetic_data(trained_generator)

# Revert normalization to original scale if needed
synthetic_features_original_scale = scaler.inverse_transform(
    synthetic_features)

# Visualize results
plt.figure(figsize=(10, 6))
plt.plot(time_series_data, label="Original Time Series")
plt.plot(synthetic_features_original_scale[0], label="Synthetic Time Series")
plt.legend()
plt.ylim([-100, 100])
plt.title("Original vs. Synthetic Time Series")
plt.savefig("validate.png", bbox_inches="tight")
