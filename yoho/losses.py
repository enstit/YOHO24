import torch.nn as nn


class YOHOLoss(nn.Module):
    def __init__(self):
        super(YOHOLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Calculate the YOHO loss for a batch of predictions and targets.

        Args:
            predictions (torch.Tensor): The predicted values from the model
                                        (batch_size, num_classes, 3).
            targets (torch.Tensor): The ground truth values
                                    (batch_size, num_classes, 3).

        Returns:
            torch.Tensor: The computed loss.
        """
        y_pred_class = predictions[:, 0::3]
        y_pred_start = predictions[:, 1::3]
        y_pred_end = predictions[:, 2::3]

        y_true_class = targets[:, 0::3]
        y_true_start = targets[:, 1::3]
        y_true_end = targets[:, 2::3]

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
