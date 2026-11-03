# Program 4 – Autoencoder


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(X_train, _), (X_test, _) = mnist.load_data()

# Normalize pixel values
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Flatten images
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Build Autoencoder
input_img = layers.Input(shape=(784,))

# Encoder
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)

# Decoder
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

# Create model
autoencoder = models.Model(input_img, decoded)

# Compile model
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Train model
autoencoder.fit(
    X_train, X_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test, X_test)
)

# Reconstruct images
decoded_imgs = autoencoder.predict(X_test)

print("Original image shape:", X_test.shape)
print("Reconstructed image shape:", decoded_imgs.shape)
print("Autoencoder training completed successfully!")
