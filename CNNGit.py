# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:04:12 2024

@author: akif_
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import h5py
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data from H5 file
class CustomDataset(Dataset):
    def __init__(self, h5_file):
        with h5py.File(h5_file, 'r') as f:
            self.inputs = torch.tensor(f['inputs'][:], dtype=torch.float32)  # Cast to float32
            self.outputs = torch.tensor(f['outputs'][:], dtype=torch.float32)  # Cast to float32

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        #self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        
        self.dropout1 = nn.Dropout(p=0.15)  # Dropout for convolutional layers
        self.dropout2 = nn.Dropout(p=0.25)  # Dropout for fully connected layer
    
        # Fully connected layer
        self.fc1 = nn.Linear(64 * 6 * 6, 20 * 3 * 2)  # Correct size to match 800 * 3 * 2 = 4800

    def forward(self, x):
        # Convolutional layers with ReLU, pooling, and dropout
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)  # Apply dropout after the first pooling layer
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)  # Apply dropout after the second pooling layer
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)  # Apply dropout after the third pooling layer
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout1(x)  # Apply dropout after the fourth pooling layer
        
        x = torch.flatten(x, 1)  # Flatten to (Batch, 64)
        
        # Fully connected layer with dropout
        x = self.fc1(x)
        x = self.dropout2(x)
        
        return x

# Instantiate the model and test
model = CNN().to(device)

summary(model, input_size=(8, 1, 500, 500))  # Input size: (batch_size, channels, height, width)
#%%
# Load the dataset

# Create a DataLoader
batch_size = 8  # You can adjust this depending on your memory capacity
initial_lr = 0.0001
lr_threshold = 0.01
lr_factor = 0.1
# Flag to check if learning rate has been reduced
lr_reduced = False
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Define a loss function and optimizer
criterion = nn.MSELoss()  # For regression, use MSELoss
optimizer = optim.Adam(model.parameters(), lr=initial_lr,weight_decay=1e-1)
#model.load_state_dict(torch.load('trained_model.pth'))

# Add variables to store losses
training_losses = []
validation_losses = []

# Load the dataset
dataset = CustomDataset('Data20Lines.h5')
train_size = int(0.9 * len(dataset))  # 80% training data
val_size = len(dataset) - train_size  # 20% validation data
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 100  # You can adjust the number of epochs

best_val_loss = float('inf')
best_model_path = 'best_model.pth'

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_train_loss = 0.0
    
    for inputs, outputs in train_loader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        outputs = outputs.to(device)
        
        predictions = model(inputs)
        outputs = outputs.squeeze(1)
        
        loss = criterion(predictions, outputs)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()
    
    # Calculate average training loss for the epoch
    avg_train_loss = running_train_loss / len(train_loader)
    training_losses.append(avg_train_loss)

    # Validation loss
    model.eval()  # Set the model to evaluation mode
    running_val_loss = 0.0
    
    with torch.no_grad():
        for inputs, outputs in val_loader:
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            
            predictions = model(inputs)
            outputs = outputs.squeeze(1)
            
            val_loss = criterion(predictions, outputs)
            running_val_loss += val_loss.item()
    
    avg_val_loss = running_val_loss / len(val_loader)
    validation_losses.append(avg_val_loss)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Best model saved at epoch {epoch + 1} with validation loss {avg_val_loss:.4f}")

    # Print the losses for this epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    
    if avg_train_loss < lr_threshold and not lr_reduced:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
        print(f"Learning rate reduced to: {param_group['lr']:.6f}")
        lr_reduced = True

# Save the training and validation losses to a text file
with open('losses_9.txt', 'w') as f:
    f.write("Epoch\tTraining Loss\tValidation Loss\n")
    for epoch, (train_loss, val_loss) in enumerate(zip(training_losses, validation_losses), 1):
        f.write(f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\n")

# Save the trained model
#torch.save(model.state_dict(), 'trained_model22.pth')
#print("Training complete. Model saved as 'trained_model.pth'.")





#%%
import numpy as np
import scipy.io as sio

model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Set the model to evaluation mode

# Assume you have a test dataset
test_dataset = CustomDataset('test_packed_data2.h5')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
all_predictions = []
all_ground_truth = []

with torch.no_grad():  # Disable gradient tracking for inference
    for inputs, outputs in test_dataloader:
        inputs = inputs.float()  # Ensure input is float32
        outputs = outputs.float()  # Ensure ground truth is float32
        inputs = inputs.to(device)
        outputs = outputs.to(device)
        # Get predictions from the model
        predictions = model(inputs)
        outputs = outputs.squeeze(1)


        
        # Append the predictions and ground truth
        all_predictions.append(predictions.cpu().numpy())
        all_ground_truth.append(outputs.cpu().numpy())

# Convert to numpy arrays
all_predictions = np.concatenate(all_predictions, axis=0)
all_ground_truth = np.concatenate(all_ground_truth, axis=0)

# Create a dictionary to store the data
data_dict = {
    'predictions': all_predictions,
    'ground_truth': all_ground_truth
}

# Save as .mat file
sio.savemat('predictions_and_ground_truth1213.mat', data_dict)

