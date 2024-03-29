import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, RandomHorizontalFlip, Grayscale, Normalize
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score

# Convenience function to create dataset
def create_dataset(root, transformation):
    dataset = ImageFolder(root, transformation)
    return dataset

# Convenience function to create data loader
def produce_loader(data, batch_size, sampler=None, shuffle=False):
    loader = torch.utils.data.DataLoader(data, batch_size = batch_size, sampler=sampler, shuffle=shuffle)
    return loader

# Viewing images
def visualize_data(dataset, figsize=(8,8), axes=3):
    indices = []
    labels_map = {
        0: "fake",
        1: "real",
    }
    cols, rows = axes, axes
    figure = plt.figure(figsize=figsize)
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        indices.append(sample_idx)
        img, label = dataset[sample_idx]
        if not (type(dataset)==list):
            img = img.swapaxes(0,1)
            img = img.swapaxes(1,2)
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img)
    print(indices)
    indices = []
    plt.show()
    
# Early stopping implementation
class EarlyStopper:
    def __init__(self, model, optimizer, patience, autoenc=None, checkpoint=True):
        # Number of iterations validation loss is allowed to increase before training is stopped
        self.patience = patience
        # Keep track of number of iterations validation loss is increasing
        self.counter = 0
        self.model = model
        self.optimizer = optimizer
        self.max_val_loss = 99999
        self.autoenc = autoenc
        self.checkpoint = checkpoint

    def early_stop(self, val_loss):
        # Determines maximum allowable validation loss
        if val_loss < self.max_val_loss:
            self.max_val_loss = val_loss
            # Reset counter
            self.counter = 0
            # Create checkpoint, if desired
            if self.checkpoint:
                name = f'./{self.model.__class__.__name__}_checkpoint'
                # If autoencoder is used, save under a different file name
                if self.autoenc:
                    name += '_withAE'
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, name)
                
        # If current validation loss exceeds previous, increment counter
        elif val_loss > self.max_val_loss:
            self.counter += 1
            # If counter exceeds patience value, stop training
            if self.counter >= self.patience:
                return True
        return False

# Testing function
def test(device, model, data_loader, criterion=nn.CrossEntropyLoss(), autoencoder=None, get_predictions=False):
    # Set model to evaluation mode
    model.eval()
    # Initialize epoch loss and accuracy
    test_loss = 0.0
    correct = 0
    total = 0
    # Get list of predictions for confusion matrix, if required
    if get_predictions:
        true_labels = torch.tensor([]).to(device)
        model_preds = torch.tensor([]).to(device)
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        # Get features from encoder part of autoencoder
        if autoencoder:
            inputs = autoencoder.get_features(inputs)
        labels = labels.to(device)
        # Compute model output and loss
        # (No grad computation here, as it is the test data)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
        # Update list of predictions for confusion matrix, if required
        if get_predictions:
            true_labels = torch.cat((true_labels, labels))
            model_preds = torch.cat((model_preds, predicted)) 
        # Accumulate loss and correct predictions for epoch
        test_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    # Calculate test loss and accuracy
    test_loss /= len(data_loader)
    test_acc = correct/total
    # Return list of predictions for confusion matrix, if required
    if get_predictions:
        true_labels = true_labels.type(torch.int64).cpu().numpy()
        model_preds = model_preds.type(torch.int64).cpu().numpy()
        print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')
        return true_labels, model_preds, test_loss, test_acc
    return test_loss, test_acc

# Training function
def train(device, model, train_loader, val_loader, optimizer, epochs, 
          criterion=nn.CrossEntropyLoss(), patience=3, autoencoder=None):
    # Performance curves data
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    # Initialise early stopper
    early_stopper = EarlyStopper(model, optimizer, patience, autoenc=autoencoder)
    
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        # Initialize epoch loss and accuracy
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_number, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Zero out gradients
            optimizer.zero_grad()
            # Get features from encoder part of autoencoder, if one is used
            if autoencoder:
                inputs = autoencoder.get_features(inputs)
            # Compute loss based on model predictions
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            # Backpropagate loss and update model weights
            loss.backward()
            optimizer.step()
            # Accumulate loss and correct predictions for epoch
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
            if (batch_number%5==0):
                print(f'Epoch {epoch+1}/{epochs}, Batch number: {batch_number}, Cumulated accuracy: {correct/total}')
        # Calculate epoch loss and accuracy
        epoch_loss /= len(train_loader)
        epoch_acc = correct/total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f'--- Epoch {epoch+1}/{epochs}: Train loss: {epoch_loss:.4f}, Train accuracy: {epoch_acc:.4f}')
        
        # Check model performance on validation set
        val_loss, val_acc = test(device, model, val_loader, criterion, autoencoder)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Save high performing models
        if val_acc >= 0.90:
            name = f'./{model.__class__.__name__}_{epoch+1}epochs'
            if autoencoder:
                name += '_withAE'
            torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, name)
        
        # Indicate when training is stopped early
        if early_stopper.early_stop(val_loss):
            print(f'--- Epoch {epoch+1}/{epochs}: Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}')
            print('Stopped early due to increasing validation loss.')
            return train_losses, train_accuracies, val_losses, val_accuracies
        
        print(f'--- Epoch {epoch+1}/{epochs}: Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}')    
    return train_losses, train_accuracies, val_losses, val_accuracies

# Confusion matrix calculations
def show_metrics(true_labels, model_preds):
    # Compute and display confusion matrix
    cm = confusion_matrix(true_labels, model_preds)
    ConfusionMatrixDisplay(cm, display_labels=['normal', 'pneumonia']).plot()
    # Compute precision (normal predicted as normal)
    print(f'Precision: {precision_score(true_labels, model_preds)}')
    # Compute recall (pneumonia predicted as pneumonia)
    print(f'Recall: {recall_score(true_labels, model_preds)}')
    # Compute f1 (combined performance based on precision and recall)
    print(f'F1 score: {f1_score(true_labels, model_preds)}')
    
# Get list of correct and wrong predictions for visualization
def get_pictures_test(device, model, data_loader, autoencoder=None):
    # Set model to evaluation mode
    model.eval()
    # Initialize lists for correct and wrong predictions
    correct_list = []
    wrong_list = []
    # Initialize tensors for true labels and model predictions
    true_labels = torch.tensor([]).to(device)
    model_preds = torch.tensor([]).to(device)
    for inputs, labels in data_loader:
        inputs_ = inputs.to(device)
        # Get features from encoder part of autoencoder, if one is used 
        if autoencoder:
            inputs_ = autoencoder.get_features(inputs_)
        labels = labels.to(device)
        # Compute model output and loss
        # (No grad computation here, as it is the test data)
        with torch.no_grad():
            outputs = model(inputs_)
            _, predicted = torch.max(outputs.data, 1)
            # Update tensors for true labels and model predictions
            true_labels = torch.cat((true_labels, labels))
            model_preds = torch.cat((model_preds, predicted))
            # Update lists for correct and wrong predictions
            if (predicted.item() == labels.item()):
                correct_list.append( (inputs.squeeze().detach().cpu().numpy(), labels.item()) )
            else:
                wrong_list.append( (inputs.squeeze().detach().cpu().numpy(), labels.item()) )
    # Send tensors to cpu and convert to numpy array for visualizing images
    true_labels = true_labels.type(torch.int64).cpu().numpy()
    model_preds = model_preds.type(torch.int64).cpu().numpy()
    return correct_list, wrong_list, true_labels, model_preds