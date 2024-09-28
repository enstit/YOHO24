import os
import sys
from pathlib import Path
import argparse
import logging
import json
import pandas as pd
import numpy as np
import torch
from torchvision.transforms import v2
from torchaudio.transforms import TimeMasking, FrequencyMasking

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOHO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
MODELS_DIR = Path(os.path.join(ROOT, "models"))

from yoho import YOHOLoss, YOHO
from yoho.utils import (
    AudioFile,
    YOHODataset,
    UrbanSEDDataset,
    YOHODataGenerator,
)

from timeit import default_timer as timer
import sed_eval
import dcase_util


def get_loss_function():
    return YOHOLoss()


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


def append_loss_dict(
    epoch, train_loss, val_loss, error_rate, f1_score, filename="losses.json"
):
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


def process_output(
    output: np.array, classes: list[str]
) -> list[tuple[str, float, float]]:

    STEPS_NO = 9
    step_duration = 2.56 / STEPS_NO
    MIN_EVENT_DURATION = 0
    MIN_SILENCE_DURATION = 1.0

    processed_output = []

    for k in range(output.shape[0]):

        labels = []
        for i in range(output.shape[2]):

            for j in range(0, output.shape[1], 3):
                if output[k, j, i] >= 0.5:
                    label = classes[j // 3]
                    start = (
                        i * step_duration
                        + output[k, j + 1, i].item() * step_duration
                    )
                    end = (
                        i * step_duration
                        + output[k, j + 2, i].item() * step_duration
                    )
                    labels.append((label, round(start, 2), round(end, 2)))

        # Order the labels by class
        labels = sorted(labels, key=lambda x: x[0])

        # Merge events of the same class that are close to each other
        merged_labels = []
        for label, start, end in labels:
            if not merged_labels:
                merged_labels.append((label, start, end))
            else:
                prev_label, prev_start, prev_end = merged_labels[-1]
                if (
                    prev_label == label
                    and start - prev_end < MIN_SILENCE_DURATION
                ):
                    merged_labels[-1] = (label, prev_start, end)
                else:
                    merged_labels.append((label, start, end))

        # Remove events that are too short
        merged_labels = [
            (label, start, end)
            for label, start, end in merged_labels
            if end - start >= MIN_EVENT_DURATION
        ]

        # Order the labels by start time
        # If two events start at the same time, order by class index
        merged_labels = sorted(
            merged_labels, key=lambda x: (x[1], classes.index(x[0]))
        )

        processed_output.append(merged_labels)

    return processed_output


def compute_metrics(predictions, targets, classes, filepaths):
    """
    Computes the error rate and F1 score for the given predictions and targets.
    """

    # Process the outputs
    processed_predictions = process_output(predictions.cpu().numpy(), classes)
    processed_targets = process_output(targets.cpu().numpy(), classes)

    temp_f1 = 0
    temp_error = 0

    total_f1_score = 0
    total_error_rate = 0

    N_events = 0

    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes,
        time_resolution=1.0,
    )

    for pred, target, filepath in zip(
        processed_predictions, processed_targets, filepaths
    ):

        if not pred and not target:
            temp_f1 += 1
            temp_error += 0
            continue

        if pred and not target:
            temp_f1 += 0
            temp_error += 1
            continue

        if not pred and target:
            temp_f1 += 0
            temp_error += 1
            N_events += len(target)
            continue

        if pred and target:
            N_events += len(target)

            # Create the event list
            pred_event_list = dcase_util.containers.MetaDataContainer(
                [
                    {
                        "file": filepath,
                        "event_label": event[0],
                        "onset": event[1],
                        "offset": event[2],
                    }
                    for event in pred
                ]
            )

            # Create the target event list
            target_event_list = dcase_util.containers.MetaDataContainer(
                [
                    {
                        "file": filepath,
                        "event_label": event[0],
                        "onset": event[1],
                        "offset": event[2],
                    }
                    for event in target
                ]
            )

            segment_based_metrics.evaluate(
                reference_event_list=target_event_list,
                estimated_event_list=pred_event_list,
            )

    overall_metrics = segment_based_metrics.results_overall_metrics()

    temp_error = temp_error / N_events
    temp_f1 = temp_f1 / N_events

    if np.isnan(overall_metrics["f_measure"]["f_measure"]):
        total_f1_score = temp_f1
    else:
        total_f1_score = overall_metrics["f_measure"]["f_measure"] + temp_f1

    total_error_rate = overall_metrics["error_rate"]["error_rate"] + temp_error

    return total_error_rate, total_f1_score


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

    criterion = get_loss_function()
    optimizer = model.get_optimizer()

    # Initialize a GradScaler
    scaler = torch.GradScaler(device=device) if autocast else None

    for epoch in range(start_epoch, num_epochs):
        # Set the model to training mode
        model.train()
        # Initialize running loss
        running_train_loss = 0.0

        start_training = timer()

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

        end_training = timer()

        # Set the model to evaluation mode
        model.eval()
        running_val_loss = 0.0
        error_rate = 0.0
        f1_score = 0.0

        # Disable gradient computation
        with torch.no_grad():

            for _, (inputs, labels) in enumerate(val_loader):

                logger.debug(
                    f"Computating metrics for observations [{_*val_loader.batch_size}:{(_ + 1)*val_loader.batch_size}] in validation dataset."
                )

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                filepaths = [
                    audio.filepath
                    for audio in val_loader.dataset.audios[
                        _
                        * val_loader.batch_size : (_ + 1)
                        * val_loader.batch_size
                    ]
                ]

                # Compute the error rate and f1 score
                running_error_rate, running_f1_score = compute_metrics(
                    predictions=outputs,
                    targets=labels,
                    classes=train_loader.dataset.labels,
                    filepaths=filepaths,
                )

                running_val_loss += loss.detach()
                error_rate += running_error_rate
                f1_score += running_f1_score

                logger.debug(
                    f"Batch metrics - Error rate: {running_error_rate:.2f}, F1-score: {running_f1_score:.2f}"
                )

            if val_loader:
                error_rate /= len(val_loader)
                f1_score /= len(val_loader)

            logger.debug(
                f"Overall metrics - Error rate: {error_rate:.2f}, F1-score: {f1_score:.2f}"
            )

            avg_val_loss = running_val_loss / len(val_loader)

        if scheduler is not None:
            # Step the scheduler (cosine annealing)
            scheduler.step()

        avg_train_loss = avg_train_loss.item()
        avg_val_loss = (
            avg_val_loss.item() if avg_val_loss is not None else None
        )

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.2f}, Val Loss: {avg_val_loss:.2f}, "
            f"Error Rate: {error_rate:.2f}, F1 Score: {f1_score:.2f}, "
            f"Time taken: {(end_training - start_training)/60:.2f} mins"
        )

        # Append the losses to the file
        append_loss_dict(
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            error_rate,
            f1_score,
            filename=model.name + "_losses.json",
        )

        # Save the model checkpoint after each epoch
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
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


def load_dataset(
    partition: str, augment: bool = False, logger: logging.Logger = None
) -> YOHODataset:

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
                        AudioFile(
                            filepath=file.filepath, labels=eval(file.events)
                        )
                        for _, file in pd.read_csv(
                            os.path.join(
                                ROOT,
                                "data/raw/UrbanSED/train.csv",
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
                ROOT, "data/processed/UrbanSED/validate.pkl"
            )

            if os.path.exists(filepath):
                logger.info(
                    "Loading the validation dataset from the pickle file"
                )
                return UrbanSEDDataset.load(filepath)

            logger.info("Creating the validation dataset")
            urbansed_val = UrbanSEDDataset(
                audios=[
                    audioclip
                    for _, audio in enumerate(
                        AudioFile(
                            filepath=file.filepath, labels=eval(file.events)
                        )
                        for _, file in pd.read_csv(
                            os.path.join(
                                ROOT,
                                "data/raw/UrbanSED/validate.csv",
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


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        default="UrbanSEDYOHO",
        help="name of the model",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=MODELS_DIR / "models/yoho.pt",
        help="model weights path",
    )

    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="training CSV path",
    )

    parser.add_argument(
        "--validate-path",
        type=str,
        default=None,
        help="validation CSV path",
    )

    parser.add_argument(
        "--classes",
        type=str,
        action="append",
        nargs="+",
        default=[],
        help="list of classes",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch size for training the model",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="maximum number of epochs to train the model",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to use for training the model",
    )

    parser.add_argument(
        "--cosine-annealing",
        action="store_true",  # default=False
        help="use Cosine Annealing learning rate scheduler",
    )

    parser.add_argument(
        "--autocast",
        action="store_true",  # default=False
        help="use autocast to reduce memory usage",
    )

    parser.add_argument(
        "--spec-augment",
        action="store_true",  # default=False
        help="augment the training data using SpecAugment library",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",  # default=False
        help="log additional information during training",
    )

    return parser.parse_args()


def main(opt: argparse.Namespace):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if opt.verbose else logging.INFO)

    logger.debug(f"Training the model for {opt.epochs} epochs")

    device = (
        opt.device if opt.device is not None else get_device(logger=logger)
    )
    logger.debug(f"Start training using device: {device}")

    # Set the seed for reproducibility
    torch.manual_seed(0)

    urbansed_train = load_dataset(partition="train", augment=opt.spec_augment)
    urbansed_val = load_dataset(partition="validate")

    logger.info("Creating the train data loader")

    # Get number of workers from slurm (default: 4)
    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK", 4))

    train_dataloader = YOHODataGenerator(
        urbansed_train,
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    logger.info("Creating the validation data loader")
    val_dataloader = YOHODataGenerator(
        urbansed_val,
        batch_size=opt.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    # Create the model
    model = YOHO(
        name=opt.name,
        input_shape=(1, 40, 257),
        n_classes=len(urbansed_train.labels),
    ).to(device)

    # Get optimizer
    optimizer = model.get_optimizer()

    scheduler = None
    if opt.cosine_annealing:  # Use cosine annealing learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.epochs
        )

    # Load the model checkpoint if it exists
    model, optimizer, start_epoch, scheduler, _ = load_checkpoint(
        model,
        optimizer,
        filename=f"{model.name}_checkpoint.pth.tar",
        scheduler=scheduler,
        logger=logger,
    )

    logger.info("Start training the model")
    start_training = timer()

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

    end_training = timer()
    seconds_elapsed = end_training - start_training
    logger.info(f"Training took {(seconds_elapsed)/60:.2f} mins")


if __name__ == "__main__":

    args = parse_arguments()
    main(opt=args)
