import torch.nn as nn


class YOHOLoss(nn.Module):
    def __init__(self):
        super(YOHOLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Calculate the YOHO loss for a batch of predictions and targets.

        Args:
            predictions (torch.Tensor): The predicted values from the model
                                        (batch_size, height, width).
            targets (torch.Tensor): The ground truth values
                                    (batch_size, height, width).

        Returns:
            torch.Tensor: The computed loss per batch.
        """

        output_class = predictions[:, 0::3, :]
        output_start = predictions[:, 1::3, :]
        output_end = predictions[:, 2::3, :]

        target_class = targets[:, 0::3, :]
        target_start = targets[:, 1::3, :]
        target_end = targets[:, 2::3, :]

        # Determine the presence of the class
        class_present = (target_class > 0).float()

        # Compute squared differences for the class, start, and end points
        error = (
            (output_class - target_class).pow(2)
            + (output_start - target_start).pow(2) * class_present
            + (output_end - target_end).pow(2) * class_present
        ).sum(dim=[1, 2])

        return error
