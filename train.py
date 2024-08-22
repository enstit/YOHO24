import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from utils import AudioFile, TUTDataset, YOHODataGenerator
from models import YOHO

class YOHOLoss(nn.Module):
    def __init__(self):
        super(YOHOLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Calculate the YOHO loss for a batch of predictions and targets.
        
        Args:
            predictions (torch.Tensor): The predicted values from the model (batch_size, num_classes, 3).
            targets (torch.Tensor): The ground truth values (batch_size, num_classes, 3).

        Returns:
            torch.Tensor: The computed loss.
        """
        y_pred_class = predictions[:, :, 0]
        y_pred_start = predictions[:, :, 1]
        y_pred_end = predictions[:, :, 2]

        y_true_class = targets[:, :, 0]
        y_true_start = targets[:, :, 1]
        y_true_end = targets[:, :, 2]

        # Compute the classification loss
        classification_loss = (y_pred_class - y_true_class).pow(2)
        # Compute the regression loss
        regression_loss = ((y_pred_start - y_true_start).pow(2) + (y_pred_end - y_true_end).pow(2)) * y_true_class
        # The total loss is the sum of the classification and regression loss
        total_loss = classification_loss + regression_loss
        return total_loss.mean()


def get_loss_function():
    return YOHOLoss()

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    if not os.path.isfile(filename):
        # If no checkpoint exists
        return model, optimizer, 0, None
    
    # Read the checkpoint file
    checkpoint = torch.load(filename)

    model.load_state_dict(
        checkpoint["state_dict"]
    )
    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    start_epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, start_epoch, loss

def train_model(model, train_loader, eval_loader, num_epochs, start_epoch=0):

    criterion = get_loss_function()
    optimizer = model.get_optimizer()
    
    for epoch in range(start_epoch, num_epochs):
        # Set the model to training mode
        model.train() 
        # Initialize running loss
        running_loss = 0.0  
        
        for _, (inputs, labels) in enumerate(train_loader):
            # Move the inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()   
            # Forward pass
            outputs = model(inputs)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Optimize the model
            optimizer.step()
            # Accumulate the loss
            running_loss += loss.item()
        
        # Compute the average loss for the epoch
        avg_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}")
        
        # Save the model checkpoint after each epoch
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss
            }
        )

        #validate_model(model, eval_loader)
        
def validate_model(model, val_loader):

    # Set the model to evaluation mode
    model.eval()

    running_loss = 0.0
    criterion = get_loss_function()
    
    # Disable gradient computation
    # And compute the loss on the validation set
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    
    print(f"Validation Loss: {running_loss / len(val_loader)}")

if __name__ == "__main__":
    print("Training YOHO model")
    # Device agnostic code
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the seed for reproducibility
    torch.manual_seed(0)
    
    print("Loading TUT dataset")
    audioclips = [
        audioclip
        for _, file in pd.read_csv("./data/tut.train.csv").iterrows()
        for audioclip in AudioFile(filepath=file.filepath, labels=file.events).audioclips(
            win_ms=2560, hop_ms=1960
        )
    ]

    # Load the TUT dataset (train)
    print("Creating dataloader")
    tut_train = TUTDataset(audioclips=audioclips) 

    train_dataloader = YOHODataGenerator(tut_train, batch_size=32, shuffle=True)

    """
    # Load the TUT dataset (evaluation)
    tut_eval = TUTDataset(
        audioclips=[
            AudioFile(filepath=file.filepath, labels=file.events)
            for _, file in pd.read_csv("./data/tut.evaluation.csv").iterrows()
        ],
    )

    eval_dataloader = YOHODataGenerator(
        dataset=tut_eval,
        batch_size=32,
        shuffle=False
    )"""

    # Define the input and output shapes
    input_shape = train_dataloader.dataset[0][0].shape
    output_shape = (train_dataloader.dataset[0][1].shape[0],)

    model = YOHO(input_shape=input_shape, output_shape=output_shape)

    # Move the model to the device
    model = model.to(device)
    
    # Get optimizer
    optimizer = model.get_optimizer()

    # Load the model checkpoint if it exists
    model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer)

    # Set the number of epochs
    EPOCHS = 10

    train_model(
        model=model, 
        train_loader=train_dataloader, 
        eval_loader=None, 
        num_epochs=EPOCHS,
        start_epoch=start_epoch
    )
