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
    YOHODataGenerator,
)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
DATA_DIR = Path(ROOT / "data" / "processed")
MODELS_DIR = Path(ROOT / "models")


def load_checkpoint(
    model: YOHO,
    optimizer: torch.optim.Optimizer,
    weights_path: Path,
    scheduler: torch.optim.lr_scheduler = None,
    logger: logging.Logger = None,
) -> tuple:

    if not os.path.exists(weights_path):
        logger.info("No checkpoint found at {weights_path}, start training from scratch")
        return model, optimizer, 0, None, None

    logger.info(f"Found checkpoint file at {weights_path}, loading checkpoint")
    checkpoint = torch.load(weights_path)

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, start_epoch, scheduler, loss


def append_to_filesystem_dict(filepath, key, **kargs):

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            dict = json.load(f)
    else:
        dict = {}

    dict[key] = {k: v for k, v in kargs.items()}

    with open(filepath, "w") as f:
        json.dump(dict, f)


def train_model(
    model: YOHO,
    device,
    train_loader,
    val_loader,
    num_epochs: int,
    start_epoch: int = 0,
    scheduler=None,
    autocast=False,
    logger: logging.Logger = None,
    losses_path: Path = None,
    weights_path: Path = None,
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

        # Set the model to evaluation mode
        model.eval()
        running_val_loss = 0.0
        # Disable gradient computation
        with torch.no_grad():
            for _, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.detach()
            avg_val_loss = running_val_loss / len(val_loader)

        if scheduler is not None:
            # Step the scheduler (cosine annealing)
            scheduler.step()

        avg_train_loss = avg_train_loss.item()
        avg_val_loss = avg_val_loss.item() if avg_val_loss is not None else None

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}]:\tTrain Loss: {avg_train_loss:.2f}\tVal Loss: {avg_val_loss:.2f}"
        )

        # Append the losses to the file
        append_to_filesystem_dict(
            filepath=losses_path, key=epoch + 1, avg_train_loss=avg_train_loss, avg_val_loss=avg_val_loss
        )

        # Save the model checkpoint after each epoch
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
                "loss": avg_train_loss,
            },
            weights_path,
        )


def load_dataset(
    filepath: Path,
    augment: bool = False,
    logger: logging.Logger = None,
    classes: list = [],
    window_size: float = 2.56,
    hop_size: float = 1.0,
) -> YOHODataset:

    # Check if exists a pickle file on the same location
    pickle_file = Path(filepath).with_suffix(".pkl")
    if pickle_file.exists():
        logger.info(f"Loading the dataset from the pickle file: {pickle_file}")
        return YOHODataset.load(pickle_file)

    logging.debug(f"Loading dataset from CSV file: {filepath}")

    if not classes:
        # Get the classes from the CSV file
        logger.debug("Unique classes for the dataset not provided, getting them from the CSV file")
        # Get the events column from the CSV file. This is a list of tuples (event, onset, offset)
        classes = pd.read_csv(filepath).events.apply(eval).explode().apply(lambda x: x[0]).unique().tolist()
        logger.debug(f"Unique classes found in the dataset: {classes}")

    # Set the data augmentation transformations, if required
    transform = None
    if augment is True:
        logger.debug("Augmenting the data using SpecAugment")
        transform = v2.Compose(
            [
                FrequencyMasking(freq_mask_param=8),
                TimeMasking(time_mask_param=25),
                TimeMasking(time_mask_param=25),
            ]
        )
    else:
        logger.debug("No augmentation applied to the data")

    dataset = YOHODataset(
        audios=[
            audioclip
            for _, audio in enumerate(
                AudioFile(filepath=file.filepath, labels=eval(file.events))
                for _, file in pd.read_csv(filepath).iterrows()
            )
            for audioclip in audio.subdivide(win_len=window_size, hop_len=hop_size)
        ],
        labels=classes,
        transform=transform,
        n_mels=40,
        hop_len=0.01,
        win_len=0.04,
    )

    # Save the dataset
    dataset.save(pickle_file)

    return dataset


def parse_arguments():

    def file_path(path):
        if os.path.isfile(path):
            return Path(path)
        else:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                return Path(path)
            except OSError:
                raise argparse.ArgumentTypeError(f"Invalid path: {path}")

    parser = argparse.ArgumentParser(description="Train YOHO model with provided arguments")

    parser.add_argument("--name", type=str, default="YOHO", help="name of the model")
    parser.add_argument("--weights-path", type=file_path, default=MODELS_DIR / "model.pt", help="model weights path")
    parser.add_argument("--losses-path", type=file_path, default=MODELS_DIR / "losses.json", help="model losses path")
    parser.add_argument("--train-path", type=str, default=DATA_DIR / "train.csv", help="training CSV path")
    parser.add_argument("--validate-path", type=str, default=DATA_DIR / "validate.csv", help="validation CSV path")
    parser.add_argument("--classes", type=str, nargs="+", default=[], help="list of classes")
    parser.add_argument("--audio-win", type=float, default=2.56, help="audio duration, in seconds, for dataset data")
    parser.add_argument("--audio-hop", type=float, default=1.00, help="audio hop size, in seconds, for dataset data")
    parser.add_argument("--mel-bands", type=int, default=40, help="number of mel bands for input spectrogram")
    parser.add_argument("--mel-win", type=float, default=0.04, help="window size, in seconds, for input spectrogram")
    parser.add_argument("--mel-hop", type=float, default=0.01, help="hop size, in seconds, for input spectrogram")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size for training the model")
    parser.add_argument("--epochs", type=int, default=50, help="maximum number of epochs to train the model")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="device to use")
    parser.add_argument("--cosine-annealing", action="store_true", help="use Cosine Annealing learning rate scheduler")
    parser.add_argument("--autocast", action="store_true", help="use autocast to reduce memory usage")
    parser.add_argument("--spec-augment", action="store_true", help="augment the training data using SpecAugment")
    parser.add_argument("--random-seed", type=int, default=None, help="random seed for reproducibility")
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
    torch.manual_seed(opt.random_seed) if opt.random_seed is not None else None

    # Set the device to train the model
    if opt.device is not None:
        device = opt.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Device not provided, using the available device: {device}")

    train_dataset = load_dataset(
        filepath=opt.train_path,
        augment=opt.spec_augment,
        logger=logger,
        classes=opt.classes,
        window_size=opt.audio_win,
        hop_size=opt.audio_hop,
    )
    val_dataset = load_dataset(
        filepath=opt.validate_path,
        logger=logger,
        classes=opt.classes,
        window_size=opt.audio_win,
        hop_size=opt.audio_hop,
    )

    train_dataloader = YOHODataGenerator(
        train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=4
    )

    val_dataloader = YOHODataGenerator(val_dataset, batch_size=opt.batch_size, pin_memory=True, num_workers=4)

    # Create the model
    model = YOHO(name=opt.name, input_shape=(1, opt.mel_bands, 257), n_classes=len(train_dataset.labels)).to(device)

    # Get optimizer
    optimizer = model.get_optimizer()

    scheduler = None
    if opt.cosine_annealing:  # Use cosine annealing learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    # Load the model checkpoint if it exists
    model, optimizer, start_epoch, scheduler, _ = load_checkpoint(
        model, optimizer, weights_path=opt.weights_path, scheduler=scheduler, logger=logger
    )

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
        losses_path=opt.losses_path,
        weights_path=opt.weights_path,
    )


if __name__ == "__main__":

    args = parse_arguments()
    main(opt=args)
