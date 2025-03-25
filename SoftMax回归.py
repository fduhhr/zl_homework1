import numpy as np
import torchvision.datasets as datasets
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# 加载 CIFAR-10 数据集
cifar10_train = datasets.CIFAR10(root='data', train=True, download=True)
cifar10_test = datasets.CIFAR10(root='data', train=False, download=True)

# 数据预处理
def preprocess_data(data):
    X = np.array(data.data) / 255.0
    X = X.reshape(X.shape[0], -1)
    y = np.array(data.targets).reshape(-1, 1)
    enc = OneHotEncoder(sparse_output=False)
    y_one_hot = enc.fit_transform(y)
    return X, y_one_hot

X_train, y_train = preprocess_data(cifar10_train)
X_test, y_test = preprocess_data(cifar10_test)
np.random.seed(42)
indices = np.random.permutation(len(X_train))
train_indices, val_indices = indices[:45000], indices[45000:]
X_val, y_val = X_train[val_indices], y_train[val_indices]
X_train, y_train = X_train[train_indices], y_train[train_indices]


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._softmax(self.z2)
        return self.a2

    def _activation(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        else:
            return z

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward(self, X, y, l2_lambda=0.0):
        m = X.shape[0]
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m + l2_lambda * self.W2 / m
        db2 = np.sum(dz2, axis=0) / m
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self._activation_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m + l2_lambda * self.W1 / m
        db1 = np.sum(dz1, axis=0) / m
        return dW1, db1, dW2, db2

    def _activation_derivative(self, z):
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            a = self._activation(z)
            return a * (1 - a)
        else:
            return np.ones_like(z)


def train(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=128, learning_rate=0.01, l2_lambda=0.01):
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_accuracy = 0.0
    best_weights = None

    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
            y_pred = model.forward(X_batch)
            loss = -np.sum(y_batch * np.log(y_pred + 1e-9)) / batch_size + 0.5 * l2_lambda * (np.sum(model.W1**2) + np.sum(model.W2**2))
            dW1, db1, dW2, db2 = model.backward(X_batch, y_batch, l2_lambda)
            model.W1 -= learning_rate * dW1
            model.b1 -= learning_rate * db1
            model.W2 -= learning_rate * dW2
            model.b2 -= learning_rate * db2

        val_loss = -np.sum(y_val * np.log(model.forward(X_val) + 1e-9)) / len(X_val) + 0.5 * l2_lambda * (np.sum(model.W1**2) + np.sum(model.W2**2))
        val_accuracy = np.mean(np.argmax(model.forward(X_val), axis=1) == np.argmax(y_val, axis=1))
        train_losses.append(loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weights = (model.W1.copy(), model.b1.copy(), model.W2.copy(), model.b2.copy())

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    model.W1, model.b1, model.W2, model.b2 = best_weights
    return train_losses, val_losses, val_accuracies


def test(model, X_test, y_test):
    accuracy = np.mean(np.argmax(model.forward(X_test), axis=1) == np.argmax(y_test, axis=1))
    print(f"Test Accuracy: {accuracy:.4f}")


def plot_curves(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

model = NeuralNetwork(input_size=3072, hidden_size=128, output_size=10)
train_losses, val_losses, val_accuracies = train(model, X_train, y_train, X_val, y_val)
plot_curves(train_losses, val_losses, val_accuracies)
test(model, X_test, y_test)

def save_weights(model, filename):
    weights = {
        'W1': model.W1,
        'b1': model.b1,
        'W2': model.W2,
        'b2': model.b2
    }
    np.savez(filename, **weights)

save_weights(model, 'model_weights.npz')