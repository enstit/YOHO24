import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    A depthwise separable convolution layer consisting of a depthwise convolution followed 
    by a pointwise convolution, batch normalization, and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
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
            stride=stride, 
            padding=kernel_size//2, 
            groups=in_channels, 
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )

        self.bn_depthwise = nn.BatchNorm2d(in_channels, eps=1e-4)
        self.bn_pointwise = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)

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
        return x

class MobileNetBackbone(nn.Module):
    """
    The MobileNet backbone consisting of an initial convolution layer followed by 
    a series of depthwise separable convolution layers.
    """
    def __init__(self, input_shape):
        """
        Initializes the MobileNet backbone.
        
        Args:
            input_shape (tuple): Shape of the input tensor (channels, height, width).
        """
        super(MobileNetBackbone, self).__init__()
        input_channels = input_shape[0]
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-4),
            nn.ReLU(inplace=True)
        )
        
        LAYER_DEFS = [
            # kernel_size, stride, out_channels
            ([3, 3], 1, 64),
            ([3, 3], 2, 128),
            ([3, 3], 1, 128),
            ([3, 3], 2, 256),
            ([3, 3], 1, 256),
            ([3, 3], 2, 512),
            ([3, 3], 1, 512),
            ([3, 3], 1, 512),
            ([3, 3], 1, 512),
            ([3, 3], 1, 512),
            ([3, 3], 1, 512),
            ([3, 3], 2, 1024),
            ([3, 3], 1, 1024)
        ]
        
        layers = []
        in_channels = 32  # Starting from the output channels of the initial convolution layer

        for kernel_size, stride, out_channels in LAYER_DEFS:
            layers.append(
                DepthwiseSeparableConv(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size[0], 
                    stride=stride
                )
            )
            in_channels = out_channels # Update the input channels for the next layer
        
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
    and final reshaping and convolution layers to match the desired output shape.
    """
    def __init__(self, input_shape, output_shape):
        """
        Initializes the YOHO model by extending the MobileNet backbone.
        
        Args:
            input_shape (tuple): Shape of the input tensor (channels, height, width).
            output_shape (tuple): Shape of the output tensor (channels, height, width).
        """
        super(YOHO, self).__init__(input_shape)
        
        self.output_shape = output_shape
        
        ADDITIONAL_LAYER_DEFS = [
            ([3, 3], 1, 512),
            ([3, 3], 1, 256),
            ([3, 3], 1, 128),
        ]
        
        layers = []
        in_channels = 1024  # Starting from the output channels of the last backbone layer
        for kernel_size, stride, out_channels in ADDITIONAL_LAYER_DEFS:
            layers.append(DepthwiseSeparableConv(in_channels, out_channels, kernel_size=kernel_size[0], stride=stride))
            in_channels = out_channels
        
        self.additional_layers = nn.Sequential(*layers)

        self.final_reshape = nn.Conv2d(128, 256, kernel_size=1)  # Adjust according to the required final reshape layer
        self.final_conv1d = nn.Conv1d(256, output_shape[0], kernel_size=1)  # Adjust according to the final Conv1D layer
        
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
        x = x.view(batch_size, 256, -1)  # Reshape to match (batch_size, channels, time)

        # Apply the final Conv1D layer
        x = self.final_conv1d(x)
        return x