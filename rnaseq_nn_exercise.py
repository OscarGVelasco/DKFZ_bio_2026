# ============================================================
# Exercise: Classifying Tumor Stage from RNA-Seq Data (PyTorch)
# Dr. Oscar Gonzalez-Velasco (oscar.gonzalezvelasco at dkfz-heidelberg.de)
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from collections import Counter

# ------------------------------------------------------------
# DATA:
# transcriptomics_df  -> RNA-seq data (samples x genes)
# clinical_df         -> clinical data with column:
#                        'tumor_stage_pathological'
# ------------------------------------------------------------

# ========================
# 0. Read the data
# ========================
# csv and tsv files - the index of patients on both tables are in column 0
transcriptomics_df = ___________________
clinical_df = ___________________

# ========================
# 1. Prepare the labels
# ========================
# Remeber always to check the integrity of the data:
Counter(clinical_df["tumor_stage_pathological"])

valid_stages = ["Stage I", "Stage II", "Stage III", "Stage IV"]

# Filter the data
clinical_df = clinical_df[
    clinical_df["tumor_stage_pathological"].isin(valid_stages)
]

# We need to transform the labels into numbers so that the NN can be trained.
# *Tip: Use scikit-learn to automatically do this and save a label-enconder to disk for reproducibility! 
stage_mapping = {
    "Stage I": 0,
    "Stage II": 1,
    "Stage III": 2,
    "Stage IV": 3
}

# TODO: Create a new column "stage_label"
clinical_df["stage_label"] = __________________________

# ========================
# 2. Match the data
# ========================
# Check that patients are both on clinical and RNA-Seq df
common_samples = transcriptomics_df.index.intersection(clinical_df.index)

# TODO: Define X and y
# These will be the main inputs of the NN
X = __________________________
y = __________________________

print("Shape of X:", X.shape)


# ========================
# 3. Train/Test split
# ========================

test_indices = []

for stage in y.unique():
    stage_idx = y[y == stage].index

    # TODO: randomly select 2 samples
    selected = __________________________

    test_indices.extend(selected)

test_indices = pd.Index(test_indices)
train_indices = X.index.difference(test_indices)

# TODO: Create splits
X_train = __________________________
X_test  = __________________________

y_train = __________________________
y_test  = __________________________


# ========================
# 4. Scale the data
# ========================
# We can NOT use the data "as-it-is", we need to scale it
# Scale the data per patient (and not by gene)
# - we want the scaling to be patient-independent
scaler = StandardScaler()

# TODO: Fit and transform
X_train_scaled = __________________________
X_test_scaled  = __________________________


# ========================
# 5. Convert to tensors
# ========================

# TODO: Convert to torch tensors
# Pytorch works with toch tensors, a different flavour of vectors
X_train_tensor = __________________________
X_test_tensor  = __________________________

y_train_tensor = __________________________
y_test_tensor  = __________________________
 

# ========================
# 6. Define the model
# ========================
# We are going to use a feed forward neuran network model with just 3 layers.
# - you have to define the number of neurons that each layer will have -> this depends on the input data
class SimpleFFNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            # TODO: First Linear layer (input -> n neurons)
            nn.Linear(input_dim, N),
            nn.ReLU(),
            # TODO: Dropout (p=probability of droping the neuron)
            nn.Dropout(p=P),
            
           # TODO: Second layer (N1 -> N2)
            __________________________,
            nn.ReLU(),

            # TODO: Output layer (N2 -> N classes)
            __________________________
        )

    def forward(self, x):
        return self.model(x)


# ========================
# 7. Setup training
# ========================
# TODO: Define the input dimension of the model (corresponds with the number of ... on the dataframe)
input_dim = _________________________
model = SimpleFFNN(input_dim)

# TODO: Define loss function
#       "A": nn.MSELoss(),
#       "B": nn.CrossEntropyLoss(),
#       "C": nn.BCELoss(), > binary cross entropy
#       "D": nn.L1Loss()  > L1 corresponds to mean absolute error (e.g. mean(y_hat - y))

criterion = __________________________

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ========================
# 8. Training loop
# ========================
# Model training stage:
# - call your model and give it the input values
# - compute the loss to understand how well the model is performing
# - use the loss to backpropagate the error and modify the weights of the model
# (REPEAT)

# TODO: define number of epochs (choose a small number if you want to have lunch at some point :) (e.g.: 10 to 30)

num_epochs = N_EPOCHS
loss_training = []

for epoch in range(num_epochs):
    model.train()

    # TODO: Forward pass
    # Use the model you have created and feed it the inputs!
    outputs = __________________________

    # TODO: Compute loss
    # Hint: use the criterion function from pytorch!
    loss = __________________________
    # TODO: Track loss data along training
    loss_training = _______________________
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# ========================
# 9. Evaluation
# ========================
# Now we move to the validation data > we will use our new model:
model.eval() # We tell pytorch that we are going into evaluation mode
            # Some elements of the model are disabled, like the dropout units

# We also tell pytoch that we do not want to keep track of the gradients:
with torch.no_grad():
    # TODO: Get predictions
    outputs = __________________________

    # TODO: Convert to predicted classes
    _, predicted = __________________________

    # TODO: Compute accuracy
    accuracy = __________________________

print(f"Test Accuracy: {accuracy.item():.4f}")

# TODO: check the outputs tensor, what are these values? why do we select the maximum?
print(outputs)

# TODO: Use the softmax function, what are these values? Has the maximum changed?
torch.softmax(outputs, dim=1)

# ========================
# Bonus Questions:
# ========================
# - Why do we scale the data?
# - Why use CrossEntropyLoss?
# - What does dropout do?
# - Is this test set reliable?
