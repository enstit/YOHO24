import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from utils import AudioFile, TUTDataset, YOHODataGenerator
from models import YOHO


def get_loss_function():
    """
    Returns the loss function to be used for training the model.
    """
    return nn.MSELoss()

def get_optimizer(model):
    """
    Returns the optimizer to be used for training the model.
    """
    return optim.Adam(model.parameters(), lr=0.001)

# Training routine
def train_model(model, train_loader, eval_loader, num_epochs):

    criterion = get_loss_function()
    optimizer = get_optimizer(model)
    
    for epoch in range(num_epochs):
        model.train() # Set the model to training mode
        running_loss = 0.0  # Initialize running loss
        
        for _, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()   
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() # Accumulate the loss
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
        
        #validate_model(model, eval_loader)
        
def validate_model(model, val_loader):

    model.eval()
    running_loss = 0.0
    criterion = get_loss_function()
    
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    
    print(f"Validation Loss: {running_loss / len(val_loader)}")

if __name__ == "__main__":

    # Device agnostic code
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Load the TUT dataset
    tut_train = TUTDataset(
        audios=[
            AudioFile(filepath=file.filepath, labels=file.events)
            for _, file in pd.read_csv("./data/tut.train.csv").iterrows()
        ], 
    )  

    train_dataloader = YOHODataGenerator(
        dataset=tut_train,
        batch_size=32,
        shuffle=True
    )

    tut_eval = TUTDataset(
        audios=[
            AudioFile(filepath=file.filepath, labels=file.events)
            for _, file in pd.read_csv("./data/tut.evaluation.csv").iterrows()
        ],
    )

    eval_dataloader = YOHODataGenerator(
        dataset=tut_eval,
        batch_size=32,
        shuffle=False
    )

    # Define the input and output shapes
    input_shape = train_dataloader.dataset[0][0].shape
    output_shape = (train_dataloader.dataset[0][1].shape[0],)

    model = YOHO(
        input_shape=input_shape, 
        output_shape=output_shape
    )
    model = model.to(device)
    

    # Train the model
    num_epochs = 10

    train_model(
        model=model, 
        train_loader=train_dataloader, 
        eval_loader=None, 
        num_epochs=num_epochs
    )
