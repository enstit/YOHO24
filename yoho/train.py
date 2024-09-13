import os
import json
import torch
import pandas as pd
from torchvision.transforms import v2
from torchaudio.transforms import TimeMasking, FrequencyMasking

from yoho import YOHOLoss, YOHO
from yoho.utils import AudioFile, UrbanSEDDataset, YOHODataGenerator

from timeit import default_timer as timer
import logging
import argparse

# import sed_eval

SCRIPT_DIRPATH = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIRPATH, "..", "models"))

logging.basicConfig(level=logging.INFO)


def get_loss_function():
    return YOHOLoss()


def save_checkpoint(state: dict, filename: str = "checkpoint.pth.tar") -> None:
    """Save the model checkpoint to a file."""
    torch.save(state, os.path.join(MODELS_DIR, filename))


def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    filepath = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(filepath):
        logging.info("No checkpoint found, starting training from scratch")
        return model, optimizer, 0, None

    logging.info(f"Found checkpoint file at {filepath}, loading checkpoint")
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, start_epoch, loss


def append_loss_dict(epoch, train_loss, val_loss, filename="losses.json"):
    filepath = os.path.join(MODELS_DIR, filename)

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


def train_model(
    model, train_loader, val_loader, num_epochs, start_epoch=0, scheduler=None
):

    criterion = get_loss_function()
    optimizer = model.get_optimizer()


    for epoch in range(start_epoch, num_epochs):
        # Set the model to training mode
        model.train()
        # Initialize running loss
        running_train_loss = 0.0

        start_epoch = timer()

        for _, (inputs, labels) in enumerate(train_loader):
            # Move the inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)
            # Forward pass
            outputs = model(inputs)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Optimize the model
            optimizer.step()
            # Accumulate the loss
            running_train_loss += loss.detach()

        logging.info("Evaluating loss") 
        # Compute the average train loss for this epoch
        avg_train_loss = running_train_loss / len(train_loader)

        end_epoch = timer()

        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0

            # ground_truth_list = []
            # prediction_list = []

            logging.info("Evaluation started")
            with torch.no_grad():
                for _, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.detach()
                avg_val_loss = running_val_loss / len(val_loader)

        else:
            avg_val_loss = None
        
        logging.info("Validation completed")

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = avg_train_loss.item()
        avg_val_loss = (
            avg_val_loss.item() if avg_val_loss is not None else None
        )

        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.2f}, Val Loss: {avg_val_loss:.2f}, "
            f"Time taken: {(end_epoch - start_epoch)/60:.2f} mins"
        )

        # Append the losses to the file
        append_loss_dict(
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            filename=model.name + "_losses.json",
        )

        # Save the model checkpoint after each epoch
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_train_loss,
            },
            model.name + "_checkpoint.pth.tar",
        )


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        # else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def load_dataset(partition: str, augment: bool = False):

    root_dir = os.path.join(SCRIPT_DIRPATH, "..")

    match partition:
        case "train":
            filepath = os.path.join(
                root_dir, "data/processed/URBAN-SED/train.pkl"
            )

            if os.path.exists(filepath):
                logging.info("Loading the train dataset from the pickle file")
                return UrbanSEDDataset.load(filepath)

            transform = None
            if augment:
                transform = v2.Compose(
                    [
                        FrequencyMasking(freq_mask_param=8),
                        TimeMasking(time_mask_param=25),
                        TimeMasking(time_mask_param=25),
                    ]
                )

            logging.info("Creating the train dataset")
            urbansed_train = UrbanSEDDataset(
                audios=[
                    audioclip
                    for _, audio in enumerate(
                        AudioFile(
                            filepath=file.filepath, labels=eval(file.events)
                        )
                        for _, file in pd.read_csv(
                            os.path.join(
                                SCRIPT_DIRPATH,
                                "../data/raw/URBAN-SED/train.csv",
                            )
                        ).iterrows()
                    )
                    for audioclip in audio.subdivide(
                        win_len=2.56, hop_len=1.00
                    )
                ],
                transform=transform,
            )

            # Save the dataset
            urbansed_train.save(filepath)
            return urbansed_train

        case "validate":

            filepath = os.path.join(
                root_dir, "data/processed/URBAN-SED/validate.pkl"
            )

            if os.path.exists(filepath):
                logging.info(
                    "Loading the validation dataset from the pickle file"
                )
                return UrbanSEDDataset.load(filepath)

            logging.info("Creating the validation dataset")
            urbansed_val = UrbanSEDDataset(
                audios=[
                    audioclip
                    for _, audio in enumerate(
                        AudioFile(
                            filepath=file.filepath, labels=eval(file.events)
                        )
                        for _, file in pd.read_csv(
                            os.path.join(
                                SCRIPT_DIRPATH,
                                "../data/raw/URBAN-SED/validate.csv",
                            )
                        ).iterrows()
                    )
                    for audioclip in audio.subdivide(
                        win_len=2.56, hop_len=1.00
                    )
                ]
            )

            # Save the dataset
            urbansed_val.save(filepath)
            return urbansed_val


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        default="UrbanSEDYOHO",
        help="The name of the model",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="The number of epochs to train the model",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size for training the model",
    )

    parser.add_argument(
        "--cosine-annealing",
        action="store_true",  # default=False
        help="Use cosine annealing learning rate scheduler",
    )

    parser.add_argument(
        "--spec-augment",
        action="store_true",  # default=False
        help="Augment the training data using SpecAugment",
    )

    args = parser.parse_args()

    if args.epochs:
        logging.info(f"Training the model for {args.epochs} epochs")

    device = get_device()
    logging.info(f"Start training using device: {device}")

    # Set the seed for reproducibility
    torch.manual_seed(0)

    urbansed_train = load_dataset(partition="train", augment=args.spec_augment)
    urbansed_val = load_dataset(partition="validate")

    logging.info("Creating the train data loader")

    # Get number of workers from slurm (default: 4)
    num_workers = int(os.getenv('SLURM_CPUS_PER_TASK', 4))

    train_dataloader = YOHODataGenerator(
        urbansed_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    logging.info("Creating the validation data loader")
    val_dataloader = YOHODataGenerator(
        urbansed_val,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    # Create the model
    model = YOHO(
        name=args.name,
        input_shape=(1, 40, 257),
        n_classes=len(urbansed_train.labels),
    ).to(device)

    # Get optimizer
    optimizer = model.get_optimizer()

    scheduler = None
    if args.cosine_annealing: # Use cosine annealing learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

    # Load the model checkpoint if it exists
    model, optimizer, start_epoch, _ = load_checkpoint(
        model, optimizer, filename=f"{model.name}_checkpoint.pth.tar"
    )

    logging.info("Start training the model")
    start_training = timer()

    # Train the model
    train_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        num_epochs=args.epochs,
        start_epoch=start_epoch,
        scheduler=scheduler,
    )

    end_training = timer()
    seconds_elapsed = end_training - start_training
    logging.info(f"Training took {(seconds_elapsed)/60:.2f} mins")
