!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip

# !unzip "UCI HAR Dataset.zip"

# pytorch_har_lstm.py

import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

# ---------------------------
# Dataset / constants (same as original)
# ---------------------------

INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

TRAIN = "train/"
TEST = "test/"
DATASET_PATH = "UCI HAR Dataset/"

def load_X(X_signals_paths):
    X_signals = []
    for signal_type_path in X_signals_paths:
        with open(signal_type_path, 'r') as file:
            # replicate your original parsing behaviour
            series = [
                np.array(row.replace('  ', ' ').strip().split(' '), dtype=np.float32)
                for row in file
            ]
        X_signals.append(series)
    # transpose to shape: (n_examples, n_steps, n_signals)
    X = np.transpose(np.array(X_signals), (1, 2, 0))
    return X

def load_y(y_path):
    with open(y_path, 'r') as file:
        y_ = np.array([row.replace('  ', ' ').strip().split(' ') for row in file], dtype=np.int32)
    return (y_ - 1).reshape(-1)  # convert to 0-based vector shape (N,)

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

# load numpy arrays
X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)
y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape, "y_test.shape:", y_test.shape)

training_data_count = len(X_train)  # 7352
test_data_count = len(X_test)       # 2947
n_steps = X_train.shape[1]          # 128
n_input = X_train.shape[2]          # 9

# ---------------------------
# Hyperparameters (matched)
# ---------------------------
n_hidden = 32
n_classes = 6

learning_rate = 0.0025
lambda_loss_amount = 0.0015   # used with tf.nn.l2_loss semantics: 0.5 * sum(param^2)
epochs = 300                  # original TF ran data*300 iterations; we run 300 passes as equivalent
batch_size = 1500
display_iter = 30000          # used similarly for printing frequency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Model (input linear + ReLU -> 2-layer LSTM -> final FC)
# ---------------------------
class HARLSTM(nn.Module):
    def __init__(self, n_input, n_hidden, n_classes, n_layers=2):
        super(HARLSTM, self).__init__()
        self.input_linear = nn.Linear(n_input, n_hidden)   # matches tf weights['hidden']
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=n_hidden, hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        # x: (batch, time, features)
        b, t, f = x.size()
        # apply input linear per-timestep
        x_flat = x.view(b * t, f)                # (b*t, n_input)
        h = self.relu(self.input_linear(x_flat)) # (b*t, n_hidden)
        h = h.view(b, t, -1)                     # (b, t, n_hidden)
        out_seq, (hn, cn) = self.lstm(h)         # out_seq: (b, t, n_hidden)
        last = out_seq[:, -1, :]                 # (b, n_hidden)
        logits = self.out(last)                  # (b, n_classes)
        return logits

model = HARLSTM(n_input, n_hidden, n_classes, n_layers=2).to(device)

# ---------------------------
# Loss, optimizer (we will add L2 as extra term like TF)
# ---------------------------
criterion = nn.CrossEntropyLoss()  # takes class indices

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ---------------------------
# Helper: l2 term like TF's lambda * sum(tf.nn.l2_loss(var))
# tf.nn.l2_loss(var) == sum(var**2) / 2
# So L2_term = lambda_loss_amount * 0.5 * sum(var**2)
# ---------------------------
def l2_regularization(model):
    l2 = 0.0
    for p in model.parameters():
        l2 = l2 + torch.sum(p.pow(2))
    return 0.5 * l2 * lambda_loss_amount

# ---------------------------
# Batch extraction that replicates original 'extract_batch_size' cycling logic
# (deterministic sequence, wraps around with modulo)
# ---------------------------
def extract_batch(X, y, global_step, batch_size):
    shape = list(X.shape)
    shape[0] = batch_size
    batch = np.empty(shape, dtype=np.float32)
    batch_y = np.empty((batch_size,), dtype=np.int64)
    total = len(X)
    for i in range(batch_size):
        index = ((global_step - 1) * batch_size + i) % total
        batch[i] = X[index]
        batch_y[i] = y[index]
    return batch, batch_y

# ---------------------------
# Training loop (mimics behaviour from original TF code)
# ---------------------------
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Pre-convert test set to tensors for faster evaluation
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

step = 1
max_steps = math.ceil((training_data_count * epochs) / batch_size)
print("Total training steps (approx):", max_steps)

# We will iterate until step > max_steps similar to TF while loop
while step <= max_steps:
    batch_x_np, batch_y_np = extract_batch(X_train, y_train, step, batch_size)
    batch_x = torch.tensor(batch_x_np, dtype=torch.float32).to(device)
    batch_y = torch.tensor(batch_y_np, dtype=torch.long).to(device)

    model.train()
    optimizer.zero_grad()
    logits = model(batch_x)                     # (batch, n_classes)
    loss = criterion(logits, batch_y) + l2_regularization(model)
    loss.backward()
    optimizer.step()

    # training accuracy for this batch
    preds = torch.argmax(logits, dim=1)
    acc = (preds == batch_y).float().mean().item()

    train_losses.append(loss.item())
    train_accuracies.append(acc)

    # printing / test eval conditions (mimic original conditions)
    if (step == 1) or ((step * batch_size) % display_iter == 0) or (step >= max_steps):
        print(f"Training iter #{step*batch_size}:   Batch Loss = {loss.item():.6f}, Accuracy = {acc:.6f}")

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_loss = criterion(test_logits, y_test_t) + l2_regularization(model)
            test_preds = torch.argmax(test_logits, dim=1)
            test_acc = (test_preds == y_test_t).float().mean().item()

        test_losses.append(test_loss.item())
        test_accuracies.append(test_acc)
        print(f"PERFORMANCE ON TEST SET: Batch Loss = {test_loss.item():.6f}, Accuracy = {test_acc:.6f}")

    step += 1

print("Optimization Finished!")

# final test evaluation (same as TF final)
model.eval()
with torch.no_grad():
    final_logits = model(X_test_t)
    final_loss = criterion(final_logits, y_test_t) + l2_regularization(model)
    final_preds = torch.argmax(final_logits, dim=1)
    final_acc = (final_preds == y_test_t).float().mean().item()

print("FINAL RESULT: Batch Loss = {:.6f}, Accuracy = {:.6f}".format(final_loss.item(), final_acc))


# Results

predictions = one_hot_predictions.argmax(1)

print("Testing Accuracy: {}%".format(100*accuracy))

print("")
print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

print("")
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)
print("Note: training and testing data is not equally distributed amongst classes, ")
print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

# Plot Results:
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix,
    interpolation='nearest',
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
