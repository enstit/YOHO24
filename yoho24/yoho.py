#!/usr/bin/env python

import torch.nn as nn
import torch.optim as optim


class DepthwiseSeparableConv(nn.Module):
    """
    A depthwise separable convolution layer consisting of a depthwise
    convolution followed by a pointwise convolution, batch normalization, and
    ReLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        dw_stride: int | tuple = 1,
        pw_stride: int | tuple = 1,
        dropout_rate: float = 0.1,
    ):
        """
        Initializes the depthwise separable convolution layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int or tuple): Stride of the convolution.
        """
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=dw_stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=pw_stride,
            #padding=1,
        )

        self.bn_depthwise = nn.BatchNorm2d(in_channels, eps=1e-4)
        self.bn_pointwise = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        Forward pass through the depthwise separable convolution layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn_pointwise(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MobileNetBackbone(nn.Module):
    """
    The MobileNet backbone consisting of an initial convolution layer followed
    by a series of depthwise separable convolution layers.
    """

    def __init__(self, input_shape: tuple, dropout_rate: float = 0.1):
        """
        Initializes the MobileNet backbone.

        Args:
            input_shape (tuple): Shape of the input tensor (channels, height,
                                 width).
        """
        super(MobileNetBackbone, self).__init__()
        input_channels = input_shape[1]

        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                input_channels,
                out_channels=32,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32, eps=1e-4),
            nn.ReLU(inplace=True),
        )

        LAYER_DEFS = [
            # kernel_size, pw_stride, dw_stride, out_channels
            ((3, 3), 1, 1, 64),
            ((3, 3), 1, 2, 128),
            ((3, 3), 1, 1, 128),
            ((3, 3), 1, 2, 256),
            ((3, 3), 1, 1, 256),
            ((3, 3), 1, 2, 512),
            ((3, 3), 1, 1, 512),
            ((3, 3), 1, 1, 512),
            ((3, 3), 1, 1, 512),
            ((3, 3), 1, 1, 512),
            ((3, 3), 1, 1, 512),
            ((3, 3), 1, 2, 1024),
            ((3, 3), 1, 1, 1024),
        ]

        layers = []
        in_channels = 32  # Starting from the output channels of the initial
        # convolution layer

        for kernel_size, pw_stride, dw_stride, out_channels in LAYER_DEFS:
            layers.append(
                DepthwiseSeparableConv(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    pw_stride=pw_stride,
                    dw_stride=dw_stride,
                    dropout_rate=dropout_rate,
                )
            )

            # Update the input channels for the next layer
            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the MobileNet backbone.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.initial_conv(x)
        
        x = self.layers(x)
        return x


class YOHO(MobileNetBackbone):
    """
    The YOHO model extending the MobileNet backbone with additional layers
    and final reshaping and convolution layers to match the desired output
    shape.
    """

    def __init__(
        self,
        input_shape: tuple,
        output_shape: tuple,
        dropout_rate: float = 0.1,
    ):
        """
        Initializes the YOHO model by extending the MobileNet backbone.

        Args:
            input_shape (tuple): Shape of the input tensor (channels, height,
                                 width).
            output_shape (tuple): Shape of the output tensor (channels, height,
                                  width).
        """
        super(YOHO, self).__init__(input_shape, dropout_rate)

        self.output_shape = output_shape

        ADDITIONAL_LAYER_DEFS = [
            ((3, 3), 1, 512),
            ((3, 3), 1, 256),
            ((3, 3), 1, 128),
        ]

        layers = []
        in_channels = 1024  # Starting from the output channels of the last

        for kernel_size, stride, out_channels in ADDITIONAL_LAYER_DEFS:
            layers.append(
                DepthwiseSeparableConv(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    pw_stride=stride,
                    dw_stride=stride,
                    dropout_rate=dropout_rate,
                )
            )

            # Update the input channels for the next layer
            in_channels = out_channels

        self.additional_layers = nn.Sequential(*layers)

        # Adjust according to the required final reshape layer
        self.final_reshape = nn.Conv2d(128, 256, kernel_size=1)

        # Adjust according to the final Conv1D layer
        self.final_conv1d = nn.Conv1d(256, output_shape[0], kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the YOHO model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Forward pass through the backbone
        x = super().forward(x)

        # Forward pass through the additional layers
        x = self.additional_layers(x)

        # Reshape the output tensor
        x = self.final_reshape(x)

        batch_size, _, sx, sy = x.shape

        # Reshape to match (batch_size, channels, time)
        x = x.view(batch_size, 256, -1)

        # Apply the final Conv1D layer
        x = self.final_conv1d(x)
        return x

    def get_optimizer(model, lr=0.001, weight_decay=0.01):
        """
        Get the optimizer for the YOHO model.

        Args:
            model (YOHO): The YOHO model.
            lr (float): Learning rate.
            weight_decay (float): Weight decay.

        Returns:
            torch.optim.Adam: The Adam optimizer
        """

        params_to_optimize = [
            # L2 on the first Conv2D layer
            {"params": model.initial_conv.parameters(), "weight_decay": 0.001},
            # L2 on subsequent Conv2D layers
            {
                "params": model.layers.parameters(),
                "weight_decay": weight_decay,
            },
            {
                "params": model.additional_layers.parameters(),
                "weight_decay": weight_decay,
            },
            {"params": model.final_reshape.parameters(), "weight_decay": 0},
            {"params": model.final_conv1d.parameters(), "weight_decay": 0},
        ]
        optimizer = optim.Adam(params_to_optimize, lr=lr)
        return optimizer
