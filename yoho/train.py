import os
import json
import torch
from torchvision.transforms import v2
from torchaudio.transforms import TimeMasking, FrequencyMasking
import pandas as pd

from yoho import YOHOLoss, YOHO
from yoho.utils import AudioFile, UrbanSEDDataset, YOHODataGenerator

from timeit import default_timer as timer
import logging

# import sed_eval

SCRIPT_DIRPATH = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIRPATH, "..", "models"))

logging.basicConfig(level=logging.INFO)


def get_loss_function():
    return YOHOLoss()


def save_checkpoint(state: dict, filename: str="checkpoint.pth.tar") -> None:
    """Save the model checkpoint to a file."""
    torch.save(
        state = state, 
        filename = os.path.join(MODELS_DIR, filename)
    )


def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    filepath = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(filepath):
        logging.info("No checkpoint found, starting training from scratch")
        return model, optimizer, 0, None

    logging.info(f"Found checkpoint file in {MODELS_DIR}, loading checkpoint")
    checkpoint = torch.load(filename)

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


def convert_to_sed_format(tensor, label_map, file_id="audio_file"):
    event_list = []
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            onset = tensor[i, j, 1].item()
            offset = tensor[i, j, 2].item()
            event_label = tensor[i, j, 0].item()

            if event_label in label_map:
                event_label_str = label_map[event_label]
                event_list.append(
                    {
                        "file": file_id,
                        "onset": onset,
                        "offset": offset,
                        "event_label": event_label_str,
                    }
                )
    return event_list


def train_model(model, train_loader, val_loader, num_epochs, start_epoch=0):

    criterion = get_loss_function()
    optimizer = model.get_optimizer()

    """# Define event labels mapping
    label_map = {
        0: "brakes squeaking",
        1: "car",
        2: "children",
        3: "large vehicle",
        4: "people speaking",
        5: "people walking",
    }

    # Set up sed_eval metrics
    evaluator = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=list(label_map.values())
    )"""

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

            # ground_truth_list = []
            # prediction_list = []

            with torch.no_grad():
                for _, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

                    """predictions = convert_to_sed_format(outputs)
                    ground_truth = convert_to_sed_format(labels)

                    ground_truth_list.extend(ground_truth)
                    prediction_list.extend(predictions)"""

            avg_val_loss = running_val_loss / len(val_loader)

            """evaluator.evaluate(
                reference_event_list=ground_truth_list,
                estimated_event_list=prediction_list,
            )

            evaluation_results = evaluator.results()"""

        else:
            avg_val_loss = None

        # Append the losses to the file
        append_loss_dict(epoch + 1, avg_loss, avg_val_loss)

        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"train loss: {avg_loss}, val Loss: {avg_val_loss}"
        )

        """print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {avg_loss}, Val Loss: {avg_val_loss}, "
            f"F1: {evaluation_results['overall']['f_measure']['f_measure']:.4f}, "
            f"ER: {evaluation_results['overall']['error_rate']['error_rate']:.4f}"
        )"""

        # Save the model checkpoint after each epoch
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            },
            model.name + "_checkpoint.pth.tar",
        )


def convert_to_sed_format(tensor, filename="audio_file"):
    event_list = []
    for i in range(tensor.shape[0]):
        onset = tensor[i, :, 1].item()
        offset = tensor[i, :, 2].item()
        event_label = tensor[i, :, 0].item()

        event_list.append(
            {
                "file": filename,
                "event_label": event_label,
                "onset": onset,
                "offset": offset,
            }
        )

    return event_list


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        # else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


if __name__ == "__main__":

    device = get_device()
    logging.info(f"Start training using device: {device}")

    # Set the seed for reproducibility
    torch.manual_seed(0)

    # Load the UrbanSED train dataset
    logging.info("Loading the train UrbanSED dataset")
    urbansed_train = UrbanSEDDataset(
        audios=[
            audioclip
            for _, audio in enumerate(
                AudioFile(filepath=file.filepath, labels=eval(file.events))
                for _, file in pd.read_csv(
                    os.path.join(
                        SCRIPT_DIRPATH, "../data/raw/URBAN-SED/train.csv"
                    )
                ).iterrows()
            )
            for audioclip in audio.subdivide(win_len=2.56, hop_len=1.00)
        ]
    )

    # Load the UrbanSED validation dataset
    logging.info("Loading the validation UrbanSED dataset")
    urbansed_val = UrbanSEDDataset(
        audios=[
            audioclip
            for _, audio in enumerate(
                AudioFile(filepath=file.filepath, labels=eval(file.events))
                for _, file in pd.read_csv(
                    os.path.join(
                        SCRIPT_DIRPATH, "../data/raw/URBAN-SED/validate.csv"
                    )
                ).iterrows()
            )
            for audioclip in audio.subdivide(win_len=2.56, hop_len=1.00)
        ]
    )

    """
    # Load the UrbanSED test dataset
    urbansed_test = UrbanSEDDataset(
        audios=[
            audioclip
            for _, audio in enumerate(
                AudioFile(filepath=file.filepath, labels=eval(file.events))
                for _, file in pd.read_csv(
                    os.path.join(
                        SCRIPT_DIRPATH, "../data/raw/URBAN-SED/test.csv"
                    )
                ).iterrows()
            )
            for audioclip in audio.subdivide(win_len=2.56, hop_len=1.00)
        ]
    )
    """

    logging.info("Creating the train data loader")
    train_dataloader = YOHODataGenerator(
        urbansed_train, batch_size=32, shuffle=True
    )

    logging.info("Creating the validation data loader")
    val_dataloader = YOHODataGenerator(
        urbansed_val, batch_size=32, shuffle=False
    )

    # Create the model
    model = YOHO(
        name="UrbanSEDYOHO",
        input_shape=(1, 40, 257),
        n_classes=len(urbansed_train.labels),
    ).to(device)

    # Get optimizer
    optimizer = model.get_optimizer()

    # Load the model checkpoint if it exists
    model, optimizer, start_epoch, _ = load_checkpoint(
        model, optimizer, "UrbanSEDYOHO_checkpoint.pth.tar"
    )

    # Set the number of epochs
    EPOCHS = 200

    logging.info("Start training the model")
    start_training = timer()

    # Train the model
    train_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        num_epochs=EPOCHS,
        start_epoch=start_epoch,
    )

    end_training = timer()
    seconds_elapsed = end_training - start_training
    logging.info(f"Training took {seconds_elapsed:.2f} seconds")
