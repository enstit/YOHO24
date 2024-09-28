#!/usr/bin/env python

import os
import sys
from pathlib import Path
import argparse
import logging
import json
import pandas as pd
import torch
from torchvision.transforms import v2
from torchaudio.transforms import TimeMasking, FrequencyMasking

from yoho import YOHOLoss, YOHO
from yoho.utils import (
    AudioFile,
    YOHODataset,
    UrbanSEDDataset,
    YOHODataGenerator,
)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
MODELS_DIR = Path(os.path.join(ROOT, "models"))


def save_checkpoint(state: dict, filename: str = "checkpoint.pth.tar") -> None:
    """Save the model checkpoint to a file."""
    torch.save(state, os.path.join(MODELS_DIR, filename))


def load_checkpoint(
    model,
    optimizer,
    filename="checkpoint.pth.tar",
    scheduler=None,
    logger: logging.Logger = None,
) -> tuple:
    filepath = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(filepath):
        logger.info("No checkpoint found, starting training from scratch")
        return model, optimizer, 0, None, None

    logger.info(f"Found checkpoint file at {filepath}, loading checkpoint")
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, start_epoch, scheduler, loss


def append_loss_dict(epoch, train_loss, val_loss, error_rate, f1_score, filename="losses.json"):
    filepath = os.path.join(MODELS_DIR, filename)

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            loss_dict = json.load(f)
    else:
        loss_dict = {}

    loss_dict[epoch] = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "error_rate": error_rate,
        "f1_score": f1_score,
    }

    with open(filepath, "w") as f:
        json.dump(loss_dict, f)


def train_model(
    model,
    device,
    train_loader,
    val_loader,
    num_epochs,
    start_epoch=0,
    scheduler=None,
    autocast=False,
    logger: logging.Logger = None,
):

    criterion = YOHOLoss()
    optimizer = model.get_optimizer()

    # Initialize a GradScaler
    scaler = torch.GradScaler(device=device) if autocast else None

    for epoch in range(start_epoch, num_epochs):
        # Set the model to training mode
        model.train()
        # Initialize running loss
        running_train_loss = 0.0

        for _, (inputs, labels) in enumerate(train_loader):
            # Move the inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

            if autocast:
                logger.debug("Using autocast to reduce memory usage")
                # Use autocast to reduce memory usage
                with torch.autocast(device_type=device):
                    # Forward pass
                    outputs = model(inputs)
                    # Compute the loss
                    loss = criterion(outputs, labels).to(device)

                # Scale the loss, call backward, and step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                # Forward pass
                outputs = model(inputs)
                # Compute the loss
                loss = criterion(outputs, labels).to(device)
                # Backward pass
                loss.backward()
                # Optimize the model
                optimizer.step()

            # Accumulate the loss
            running_train_loss += loss.detach()

        # Compute the average train loss for this epoch
        avg_train_loss = running_train_loss / len(train_loader)

        if scheduler is not None:
            # Step the scheduler (cosine annealing)
            scheduler.step()

        avg_train_loss = avg_train_loss.item()
        avg_val_loss = avg_val_loss.item() if avg_val_loss is not None else None

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}]:\tTrain Loss: {avg_train_loss:.2f}\tVal Loss: {avg_val_loss:.2f}"
        )

        # Append the losses to the file
        append_loss_dict(epoch + 1, avg_train_loss, avg_val_loss, filename=model.name + "_losses.json")

        # Save the model checkpoint after each epoch
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
                "loss": avg_train_loss,
            },
            model.name + "_checkpoint.pth.tar",
        )


def get_device(logger: logging.Logger = None) -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        # else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def load_dataset(partition: str, augment: bool = False, logger: logging.Logger = None) -> YOHODataset:

    match partition:
        case "train":
            filepath = os.path.join(ROOT, "data/processed/UrbanSED/train.pkl")

            if os.path.exists(filepath):
                logger.info("Loading the train dataset from the pickle file")
                return UrbanSEDDataset.load(filepath)

            transform = None
            if augment:
                logger.info("Augmenting the training data using SpecAugment")
                transform = v2.Compose(
                    [
                        FrequencyMasking(freq_mask_param=8),
                        TimeMasking(time_mask_param=25),
                        TimeMasking(time_mask_param=25),
                    ]
                )

            logger.info("Creating the train dataset")
            urbansed_train = UrbanSEDDataset(
                audios=[
                    audioclip
                    for _, audio in enumerate(
                        AudioFile(filepath=file.filepath, labels=eval(file.events))
                        for _, file in pd.read_csv(os.path.join(ROOT, "data/raw/UrbanSED/train.csv")).iterrows()
                    )
                    for audioclip in audio.subdivide(win_len=2.56, hop_len=1.00)
                ],
                transform=transform,
            )

            # Save the dataset
            urbansed_train.save(filepath)
            return urbansed_train

        case "validate":

            filepath = os.path.join(ROOT, "data/processed/UrbanSED/validate.pkl")

            if os.path.exists(filepath):
                logger.info("Loading the validation dataset from the pickle file")
                return UrbanSEDDataset.load(filepath)

            logger.info("Creating the validation dataset")
            urbansed_val = UrbanSEDDataset(
                audios=[
                    audioclip
                    for _, audio in enumerate(
                        AudioFile(filepath=file.filepath, labels=eval(file.events))
                        for _, file in pd.read_csv(os.path.join(ROOT, "data/raw/UrbanSED/validate.csv")).iterrows()
                    )
                    for audioclip in audio.subdivide(win_len=2.56, hop_len=1.00)
                ]
            )

            # Save the dataset
            urbansed_val.save(filepath)
            return urbansed_val


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="UrbanSEDYOHO", help="name of the model")
    parser.add_argument("--weights", type=str, default=MODELS_DIR / "models/yoho.pt", help="model weights path")
    parser.add_argument("--train-path", type=str, default=None, help="training CSV path")
    parser.add_argument("--validate-path", type=str, default=None, help="validation CSV path")
    parser.add_argument("--classes", type=str, action="append", nargs="+", default=[], help="list of classes")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size for training the model")
    parser.add_argument("--epochs", type=int, default=50, help="maximum number of epochs to train the model")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="device to use")
    parser.add_argument("--cosine-annealing", action="store_true", help="use Cosine Annealing learning rate scheduler")
    parser.add_argument("--autocast", action="store_true", help="use autocast to reduce memory usage")
    parser.add_argument("--spec-augment", action="store_true", help="augment the training data using SpecAugment")
    parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="log additional information during training")

    return parser.parse_args()


def main(opt: argparse.Namespace):

    # Set up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if opt.verbose else logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.handlers[0].setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.info(f"Logging level set to {logging.getLevelName(logger.getEffectiveLevel())}")

    # Log all the arguments with their values
    logger.debug("Arguments for the training:")
    for arg, value in vars(opt).items():
        logger.debug(f"\t{arg}: {value}")

    # Set the seed for reproducibility
    torch.manual_seed(opt.seed)

    device = opt.device if opt.device is not None else get_device(logger=logger)

    urbansed_train = load_dataset(partition="train", augment=opt.spec_augment, logger=logger)
    urbansed_val = load_dataset(partition="validate", logger=logger)

    logger.info("Creating the train data loader")

    # Get number of workers from slurm (default: 4)
    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK", 4))

    train_dataloader = YOHODataGenerator(
        urbansed_train, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=num_workers
    )

    logger.info("Creating the validation data loader")
    val_dataloader = YOHODataGenerator(
        urbansed_val, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=num_workers
    )

    # Create the model
    model = YOHO(name=opt.name, input_shape=(1, 40, 257), n_classes=len(urbansed_train.labels)).to(device)

    # Get optimizer
    optimizer = model.get_optimizer()

    scheduler = None
    if opt.cosine_annealing:  # Use cosine annealing learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    # Load the model checkpoint if it exists
    model, optimizer, start_epoch, scheduler, _ = load_checkpoint(
        model, optimizer, filename=f"{model.name}_checkpoint.pth.tar", scheduler=scheduler, logger=logger
    )

    logger.info("Start training the model")

    # Train the model
    train_model(
        model=model,
        device=device,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        num_epochs=opt.epochs,
        start_epoch=start_epoch,
        scheduler=scheduler,
        autocast=opt.autocast,
        logger=logger,
    )


if __name__ == "__main__":

    args = parse_arguments()
    main(opt=args)
