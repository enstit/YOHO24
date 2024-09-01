import os
import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchaudio.transforms import TimeMasking, FrequencyMasking
import pandas as pd
from utils import AudioFile, TUTDataset, YOHODataGenerator
from yoho import YOHO
import json

SCRIPT_DIRPATH = os.path.abspath(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(SCRIPT_DIRPATH, "..", "reports")


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
        regression_loss = (
            (y_pred_start - y_true_start).pow(2)
            + (y_pred_end - y_true_end).pow(2)
        ) * y_true_class
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

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, start_epoch, loss


def append_loss_dict(epoch, train_loss, val_loss, filename="losses.json"):
    filepath = os.path.join(REPORTS_DIR, filename)

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            loss_dict = json.load(f)
    else:
        loss_dict = {}

    loss_dict[epoch] = {
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    with open(filepath, "w") as f:
        json.dump(loss_dict, f)


def train_model(model, train_loader, val_loader, num_epochs, start_epoch=0):

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

        # Compute the average loss for this epoch
        avg_loss = running_loss / len(train_loader)

        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for _, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
            avg_val_loss = running_val_loss / len(val_loader)
        else:
            avg_val_loss = None

        # Append the losses to the file
        append_loss_dict(epoch + 1, avg_loss, avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}, Val Loss: {avg_val_loss}"
        )

        # Save the model checkpoint after each epoch
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            }
        )


if __name__ == "__main__":

    # Device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the seed for reproducibility
    torch.manual_seed(0)

    training_audioclips = [
        audioclip
        for _, file in pd.read_csv(
            os.path.join(
                SCRIPT_DIRPATH,
                "../data/processed/TUT/TUT-sound-events-2017-development.csv",
            )
        ).iterrows()
        for audioclip in AudioFile(
            filepath=file.filepath, labels=eval(file.events)
        ).subdivide(win_len=2.56, hop_len=1.96)
    ]

    evaluation_audioclips = [
        audioclip
        for _, file in pd.read_csv(
            os.path.join(
                SCRIPT_DIRPATH,
                "../data/processed/TUT/TUT-sound-events-2017-evaluation.csv",
            )
        ).iterrows()
        for audioclip in AudioFile(
            filepath=file.filepath, labels=eval(file.events)
        ).subdivide(win_len=2.56, hop_len=1.96)
    ]

    transforms = v2.Compose(
        [
            TimeMasking(time_mask_param=25),
            TimeMasking(time_mask_param=25),
            FrequencyMasking(freq_mask_param=8),
        ]
    )

    train_dataloader = YOHODataGenerator(
        dataset=TUTDataset(
            audios=training_audioclips,
            transform=transforms,
        ),
        batch_size=32,
        shuffle=True,
    )

    eval_dataloader = YOHODataGenerator(
        dataset=TUTDataset(audios=evaluation_audioclips),
        batch_size=32,
        shuffle=False,
    )

    model = YOHO(input_shape=(1, 40, 257), n_classes=6)
    print(model.output_channels)

    # Move the model to the device
    model = model.to(device)

    # Get optimizer
    optimizer = model.get_optimizer()

    # Load the model checkpoint if it exists
    model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer)

    # Set the number of epochs
    EPOCHS = 60

    train_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=eval_dataloader,
        num_epochs=EPOCHS,
        start_epoch=start_epoch,
    )
