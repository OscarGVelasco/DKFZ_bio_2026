# ============================================================
# SOLUTION: Classifying Tumor Stage from RNA-Seq Data (PyTorch)
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

# ========================
# 0. Read the data
# ========================
# csv and tsv files - the index of patients on both tables are in column 0
transcriptomics_df = pd.read_csv("RNA_seq_counts_Colon_adenocarcinoma.csv", index_col=0)
clinical_df = pd.read_csv("clinical_dataframe_Colon_adenocarcinoma.tsv", sep="\t",  index_col=0)


# ========================
# 1. Prepare the labels
# ========================

valid_stages = ["Stage I", "Stage II", "Stage III", "Stage IV"]

clinical_df = clinical_df[
    clinical_df["tumor_stage_pathological"].isin(valid_stages)
]

stage_mapping = {
    "Stage I": 0,
    "Stage II": 1,
    "Stage III": 2,
    "Stage IV": 3
}

clinical_df["stage_label"] = clinical_df["tumor_stage_pathological"].map(stage_mapping)


# ========================
# 2. Match the data
# ========================

common_samples = transcriptomics_df.index.intersection(clinical_df.index)

X = transcriptomics_df.loc[common_samples]
y = clinical_df.loc[common_samples, "stage_label"]

print("Shape of X:", X.shape)


# ========================
# 3. Train/Test split
# ========================

test_indices = []

for stage in y.unique():
    stage_idx = y[y == stage].index
    selected = np.random.choice(stage_idx, size=2, replace=False)
    test_indices.extend(selected)

test_indices = pd.Index(test_indices)
train_indices = X.index.difference(test_indices)

X_train = X.loc[train_indices]
X_test  = X.loc[test_indices]

y_train = y.loc[train_indices]
y_test  = y.loc[test_indices]


# ========================
# 4. Scale the data
# ========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ========================
# 5. Convert to tensors
# ========================

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.long)


# ========================
# 6. Define the model
# ========================

class SimpleFFNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1064),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1064, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.model(x)


# ========================
# 7. Setup training
# ========================

input_dim = X_train_tensor.shape[1]
model = SimpleFFNN(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ========================
# 8. Training loop
# ========================

num_epochs = 10
loss_training = []

for epoch in range(num_epochs):
    model.train()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss_training.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# ========================
# 9. Evaluation
# ========================

model.eval() # We tell pytorch that we are going into evaluation mode
            # Some elements of the model are disabled, like the dropout units

# We also tell pytoch that we do not want to keep track of the gradients:
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == y_test_tensor).float().mean()

print(f"Test Accuracy: {accuracy.item():.4f}")
