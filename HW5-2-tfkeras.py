import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Step 1: Business Understanding
# Goal: Build a Dense NN and CNN to identify handwritten digits using the MNIST dataset.

# Step 2: Data Understanding
# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# Visualize some examples
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()

# Step 3: Data Preparation
# Normalize the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape data for CNN
X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test_cnn = X_test.reshape(X_test.shape[0], 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 4: Modeling
# Dense NN Model
def create_dense_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# CNN Model
def create_cnn_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 5: Evaluation
# Train and evaluate Dense NN
print("Training Dense Neural Network...")
dense_model = create_dense_model()
dense_history = dense_model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Train and evaluate CNN
print("Training Convolutional Neural Network...")
cnn_model = create_cnn_model()
cnn_history = cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Step 6: Evaluation and Deployment
# Evaluate on test set
dense_test_loss, dense_test_acc = dense_model.evaluate(X_test, y_test, verbose=0)
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)

print(f"Dense NN Test Accuracy: {dense_test_acc * 100:.2f}%")
print(f"CNN Test Accuracy: {cnn_test_acc * 100:.2f}%")

# Visualize training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(dense_history.history['accuracy'], label='Dense NN Train Accuracy')
plt.plot(dense_history.history['val_accuracy'], label='Dense NN Val Accuracy')
plt.title('Dense NN Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['accuracy'], label='CNN Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Val Accuracy')
plt.title('CNN Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
