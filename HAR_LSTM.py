# pytorch_har_lstm_warmup_adamw_cosine.py
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn import metrics

# ---------------------------
# Dataset / constants
# ---------------------------
INPUT_SIGNAL_TYPES = [
    "body_acc_x_","body_acc_y_","body_acc_z_",
    "body_gyro_x_","body_gyro_y_","body_gyro_z_",
    "total_acc_x_","total_acc_y_","total_acc_z_"
]

LABELS = [
    "WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS",
    "SITTING","STANDING","LAYING"
]

TRAIN = "train/"
TEST = "test/"
DATASET_PATH = "UCI HAR Dataset/"

def load_X(X_signals_paths):
    X_signals = []
    for signal_type_path in X_signals_paths:
        with open(signal_type_path, 'r') as file:
            series = [np.array(row.replace('  ', ' ').strip().split(' '), dtype=np.float32) for row in file]
        X_signals.append(series)
    X = np.transpose(np.array(X_signals), (1, 2, 0))
    return X

def load_y(y_path):
    with open(y_path, 'r') as file:
        y_ = np.array([row.replace('  ', ' ').strip().split(' ') for row in file], dtype=np.int32)
    return (y_ - 1).reshape(-1)

X_train_signals_paths = [DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]
X_test_signals_paths  = [DATASET_PATH + TEST  + "Inertial Signals/" + signal + "test.txt"  for signal in INPUT_SIGNAL_TYPES]

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path  = DATASET_PATH + TEST  + "y_test.txt"

X_train = load_X(X_train_signals_paths)
X_test  = load_X(X_test_signals_paths)
y_train = load_y(y_train_path)
y_test  = load_y(y_test_path)

print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape, "y_test.shape:", y_test.shape)

training_data_count = len(X_train)  # 7352
test_data_count = len(X_test)       # 2947
n_steps = X_train.shape[1]          # 128
n_input = X_train.shape[2]          # 9

# ---------------------------
# Hyperparameters
# ---------------------------
n_hidden = 64
n_classes = 6

learning_rate = 0.0025
weight_decay_val = 1e-4     # AdamW weight decay (try 1e-4)
epochs = 300
batch_size = 1500
display_iter = 30000

# warmup + scheduler params
warmup_steps = 1000          # try 500..2000
warmup_start_lr = 1e-6
eta_min = 1e-5               # cosine minimum LR
grad_clip = 5.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Model
# ---------------------------
class HARLSTM(nn.Module):
    def __init__(self, n_input, n_hidden, n_classes, n_layers=2):
        super(HARLSTM, self).__init__()
        self.input_linear = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=n_hidden, hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        b, t, f = x.size()
        x_flat = x.view(b * t, f)
        h = self.relu(self.input_linear(x_flat))
        h = h.view(b, t, -1)
        out_seq, (hn, cn) = self.lstm(h)
        last = out_seq[:, -1, :]
        logits = self.out(last)
        return logits

model = HARLSTM(n_input, n_hidden, n_classes, n_layers=2).to(device)

# ---------------------------
# Improved initialization
# ---------------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        if m is model.input_linear:
            init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None: init.zeros_(m.bias)
        elif m is model.out:
            init.xavier_uniform_(m.weight)
            if m.bias is not None: init.zeros_(m.bias)
        else:
            init.xavier_uniform_(m.weight)
            if m.bias is not None: init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                sz = param.data.size()
                if len(sz) == 2 and sz[0] % 4 == 0 and sz[0] // 4 == sz[1]:
                    H = sz[1]
                    for i in range(4):
                        init.orthogonal_(param.data[i*H:(i+1)*H])
                else:
                    init.orthogonal_(param.data)
            elif 'bias' in name:
                init.zeros_(param.data)
                hidden_size = model.lstm.hidden_size
                if param.data.numel() == 4 * hidden_size:
                    param.data[hidden_size:2*hidden_size].fill_(1.0)

# reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

model.apply(init_weights)

# ---------------------------
# Loss, optimizer, scheduler (warmup handled manually)
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_val)

max_steps = math.ceil((training_data_count * epochs) / batch_size)
cosine_T_max = max(1, max_steps - warmup_steps)   # cosine covers remainder after warmup
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_T_max, eta_min=eta_min)

# helper warmup routine (linear)
def apply_warmup_lr(step):
    # step is 1-based global step in our loop
    if step <= warmup_steps:
        # linear ramp from warmup_start_lr to learning_rate
        lr = warmup_start_lr + (learning_rate - warmup_start_lr) * (step / float(max(1, warmup_steps)))
        for g in optimizer.param_groups:
            g['lr'] = lr
        return True
    return False

# ---------------------------
# Batch extraction (same wrap-around)
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
# Training loop (records metrics + LR)
# ---------------------------
train_losses, train_accuracies, train_steps = [], [], []
test_losses, test_accuracies, test_steps = [], [], []
lrs, lr_steps = [], []

X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

step = 1
print("Total training steps (approx):", max_steps)

while step <= max_steps:
    batch_x_np, batch_y_np = extract_batch(X_train, y_train, step, batch_size)
    batch_x = torch.tensor(batch_x_np, dtype=torch.float32).to(device)
    batch_y = torch.tensor(batch_y_np, dtype=torch.long).to(device)

    model.train()
    optimizer.zero_grad()
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    loss.backward()

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    # scheduler or warmup update
    if apply_warmup_lr(step):
        # warmup sets optimizer.lr directly
        pass
    else:
        # step scheduler AFTER first non-warmup update
        # Note: scheduler.step() updates lr for next step; we call it each step after warmup
        scheduler.step()

    # record current LR
    current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)
    lr_steps.append(step * batch_size)

    preds = torch.argmax(logits, dim=1)
    acc = (preds == batch_y).float().mean().item()

    current_iter = step * batch_size
    train_losses.append(loss.item())
    train_accuracies.append(acc)
    train_steps.append(current_iter)

    # Evaluate occasionally like original script
    if (step == 1) or ((current_iter) % display_iter == 0) or (step >= max_steps):
        print(f"Training iter #{current_iter}:   Batch Loss = {loss.item():.6f}, Acc = {acc:.6f}, LR = {current_lr:.3e}")

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_loss = criterion(test_logits, y_test_t)
            test_preds = torch.argmax(test_logits, dim=1)
            test_acc = (test_preds == y_test_t).float().mean().item()

        test_losses.append(test_loss.item())
        test_accuracies.append(test_acc)
        test_steps.append(current_iter)

        print(f"PERFORMANCE ON TEST SET: Batch Loss = {test_loss.item():.6f}, Accuracy = {test_acc:.6f}")

    step += 1

print("Optimization Finished!")

# ---------------------------
# Final test evaluation
# ---------------------------
model.eval()
with torch.no_grad():
    final_logits = model(X_test_t)
    final_loss = criterion(final_logits, y_test_t)
    final_preds = torch.argmax(final_logits, dim=1)
    final_acc = (final_preds == y_test_t).float().mean().item()

print("FINAL RESULT: Batch Loss = {:.6f}, Accuracy = {:.6f}".format(final_loss.item(), final_acc))

# ---------------------------
# Plotting: losses/accuracies + LR
# ---------------------------
font = {'family':'Bitstream Vera Sans','weight':'bold','size':16}
import matplotlib
matplotlib.rc('font', **font)

plt.figure(figsize=(12, 12))
plt.plot(train_steps, train_losses, "b--", label="Train losses")
plt.plot(train_steps, train_accuracies, "g--", label="Train accuracies")
if len(test_steps) > 0:
    plt.plot(test_steps, test_losses, "b-", label="Test losses")
    plt.plot(test_steps, test_accuracies, "g-", label="Test accuracies")
plt.title("Training progress")
plt.xlabel("Training iteration")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.show()

# LR plot
plt.figure(figsize=(10,5))
plt.plot(lr_steps, lrs, label="learning rate")
plt.title("Learning rate schedule (warmup + cosine)")
plt.xlabel("Training iteration")
plt.ylabel("LR")
plt.grid(True)
plt.show()

# ---------------------------
# Metrics & confusion matrix
# ---------------------------
final_logits_cpu = final_logits.detach().cpu().numpy()
predictions = np.argmax(final_logits_cpu, axis=1)
y_true = np.array(y_test).reshape(-1)

print("Testing Accuracy: {:.2f}%".format(100.0 * metrics.accuracy_score(y_true, predictions)))
print("Precision: {:.2f}%".format(100.0 * metrics.precision_score(y_true, predictions, average="weighted")))
print("Recall: {:.2f}%".format(100.0 * metrics.recall_score(y_true, predictions, average="weighted")))
print("F1 score: {:.2f}%".format(100.0 * metrics.f1_score(y_true, predictions, average="weighted")))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, predictions)
print("Confusion Matrix:\n", cm)
cm_norm = cm.astype(np.float32) / cm.sum() * 100.0
plt.figure(figsize=(10,10))
plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.rainbow)
plt.title("Confusion matrix (% of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
