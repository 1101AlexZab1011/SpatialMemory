import math
import torch, torch.nn as nn, torch.nn.functional as F
from utils.diffwave.layers import ResidualBlock

from utils.diffwave.utils import DiffusionParams, ResidualParams


class DiffWave(nn.Module):
    """
    PyTorch Module for a DiffWave model implementing diffusion probabilistic models.

    Args:
        input_channels (int): Number of input channels.
        diffusion_params (DiffusionParams): Parameters for the diffusion process.
        residual_params (ResidualParams): Parameters for the residual blocks.

    Attributes:
        input_channels (int): Number of input channels for the model.
        diffusion_params (DiffusionParams): Parameters for the diffusion process.
        residual_params (ResidualParams): Parameters for the residual blocks.
        diffusion_embedding: The diffusion embedding to be used in the model.
        spectrogram_upsampler: Upsampler for the input spectrogram.
        input_projection (Conv1d): Input projection layer.
        output_projection (Conv1d): Output projection layer.
        residual_layers (ModuleList): List of residual blocks in the model.
        skip_projection (Conv1d): Projection for skip connection.

    Methods:
        forward(input_, diffusion_step, spectrogram=None): Performs a forward pass through the DiffWave model.
    """
    def __init__(
        self,
        input_channels: int,
        diffusion_params: DiffusionParams,
        residual_params: ResidualParams
    ):
        super().__init__()
        self.input_channels = input_channels
        self.diffusion_params = diffusion_params
        self.residual_params = residual_params

        self.diffusion_embedding = self.diffusion_params.embedding
        self.spectrogram_upsampler = None if self.residual_params.conditioner is None\
            else self.residual_params.conditioner.upsampler

        self.input_projection = nn.Conv1d(
            self.input_channels,
            self.residual_params.n_residual_channels,
            1
        )
        self.output_projection = nn.Conv1d(
            self.residual_params.n_residual_channels,
            self.input_channels,
            1
        )
        nn.init.zeros_(self.output_projection.weight) # zero init for output projection

        if self.residual_params.conditioner is not None:
            n_mels = self.residual_params.conditioner.n_mels
            n_channels = self.residual_params.conditioner.n_channels
        else:
            n_mels = None
            n_channels = 1

        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                self.residual_params.n_residual_channels,
                2**(i % self.residual_params.dilation_cycle_length),
                self.residual_params.kernel,
                self.diffusion_params.diffusion_dim,
                n_mels, n_channels
            )
            for i in range(self.residual_params.n_residual_layers)
        ])

        self.skip_projection = nn.Conv1d(
            self.residual_params.n_residual_channels,
            self.residual_params.n_residual_channels,
            1
        )

    def forward(
        self,
        input_: torch.Tensor,
        diffusion_step: int | float,
        spectrogram: torch.Tensor = None
    ):
        """
        Performs a forward pass through the DiffWave model.

        Args:
            input_ (torch.Tensor): Input tensor to the model.
            diffusion_step (int | float): Tensor representing the diffusion step.
            spectrogram (torch.Tensor, optional): Spectrogram tensor (default is None).

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        if (spectrogram is None) != (self.spectrogram_upsampler is None):
            if (spectrogram is None):
                raise ValueError('Conditioner is required for conditional model')
            else:
                raise ValueError('Conditioner is provided, but the model is unconditional')


        x = input_.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        if self.spectrogram_upsampler: # use conditional model
            spectrogram = self.spectrogram_upsampler(spectrogram)
            spectrogram = torch.permute(spectrogram, (0, 2, 1, 3))

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x