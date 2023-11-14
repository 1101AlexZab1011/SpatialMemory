from math import sqrt
import torch, torch.nn as nn

class DiffusionEmbedding(nn.Module):
    """
    PyTorch Module for performing Diffusion Embedding based on specified parameters.

    Implements an architecture that utilizes a learned embedding for diffusion steps.

    Args:
        max_steps (int): The maximum number of diffusion steps.
        embedding_dim (int): The dimension of the diffusion embedding (default is 64).
        diffusion_dim (int): The dimension of the diffusion process (default is 512).

    Attributes:
        max_steps (int): The maximum number of diffusion steps.
        embedding_dim (int): The dimension of the diffusion embedding.
        diffusion_dim (int): The dimension of the diffusion process.
        embedding (Tensor): The precomputed embedding for diffusion steps.
        input_projection (Linear): Input projection layer.
        output_projection (Linear): Output projection layer.

    Methods:
        _build_embedding(): Builds the diffusion embedding table based on the max_steps and embedding_dim.
        _lerp_embedding(t: float): Performs linear interpolation for the diffusion embedding at a specific time step.
        forward(diffusion_step: int | float): Forward pass through the diffusion embedding model.

    References:
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).
        "Attention is All you Need." arXiv (Cornell University), 30, 5998–6008.
        [https://arxiv.org/pdf/1706.03762v5](https://arxiv.org/pdf/1706.03762v5)
    """
    def __init__(
        self,
        max_steps: int,
        embedding_dim: int = 64,
        diffusion_dim: int = 512,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.embedding_dim = embedding_dim
        self.diffusion_dim = diffusion_dim
        self.register_buffer('embedding', self._build_embedding(), persistent=False)
        self.input_projection = nn.Linear(self.embedding_dim*2, self.diffusion_dim)
        self.output_projection = nn.Linear(self.diffusion_dim, self.diffusion_dim)

    def _build_embedding(self):
        """
        Builds the diffusion embedding based on the maximum steps and embedding dimension.
        Generates the embedding table using sinusoidal functions.
        """
        steps = torch.arange(self.max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(self.embedding_dim).unsqueeze(0)          # [1,D_e]
        table = steps * 10.0**(dims * 4.0 / (self.embedding_dim - 1))     # [T,D_e]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

    def _lerp_embedding(self, t: float):
        """
        Performs linear interpolation for the diffusion embedding at a specific time step (t).
        Utilizes the precomputed embedding table to interpolate values for non-integer time steps.
        """
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def forward(self, diffusion_step: int | float):
        """
        Forward pass through the diffusion embedding model.

        Args:
            diffusion_step (int | float): The diffusion step for which to compute the embedding.

        Returns:
            Tensor: The output tensor resulting from the diffusion embedding process.
        """
        if isinstance(diffusion_step, (int, torch.int32, torch.int64)):
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.input_projection(x)
        x = nn.functional.silu(x)
        x = self.output_projection(x)
        x = nn.functional.silu(x)
        return x


class SpectrogramUpsampler(nn.Module):
    """
    PyTorch Module for upsampling spectrogram data using transpose convolutional layers.

    Args:
        n_channels (int): Number of input and output channels (default is 1).
        kernel_size (tuple[int, int]): Size of the convolutional kernel (default is (3, 32)).
        stride (tuple[int, int]): Stride value for the convolutional operation (default is (1, 16)).
        padding (tuple[int, int]): Padding applied to the input tensor (default is (1, 8)).
        negative_slope (float): Slope value for the LeakyReLU activation (default is 0.4).
        n_layers (int): Number of upsampling layers (default is 2).

    Attributes:
        upsampler (Sequential): Sequential module consisting of ConvTranspose2d layers followed by LeakyReLU.

    Methods:
        forward(x): Performs a forward pass through the upsampler module.

    """
    def __init__(
        self,
        n_channels: int = 1,
        kernel_size: tuple[int, int] = (3, 32),
        stride: tuple[int, int] = (1, 16),
        padding: tuple[int, int] = (1, 8),
        negative_slope=0.4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.upsampler = nn.Sequential(
            *(
                nn.Sequential(
                    nn.ConvTranspose2d(n_channels, n_channels, kernel_size, stride, padding),
                    nn.LeakyReLU(negative_slope),
                ) for _ in range(n_layers)
            )
        )

    def forward(self, x: torch.Tensor):
        """
        Performs an upsampling operation on the input tensor using transpose convolutional layers.

        Args:
            x (Tensor): Input tensor representing the spectrogram data.
                        Shape should be n_batch x n_channels x n_mel_frames x n_mel_bins.

        Returns:
            Tensor: Output tensor after the upsampling operation.
        """
        # x.shape ~ n_batch x n_channels x n_mel_frames x n_mel_bins
        x = self.upsampler(x)
        return x


class ResidualBlock(nn.Module):
    """
    PyTorch Module for a residual block in a DiffWave model.

    Args:
        n_residual_channels (int): Number of residual channels in the block.
        dilation (int): Dilation value for the convolutional layers.
        kernel (int, optional): Kernel size for dilated convolution (default is 3).
        diffusion_dim (int, optional): Dimension for diffusion projection (default is 512).
        n_mels (int, optional): Number of Mel-spectrogram channels. If None, unconditional model is used (default is None).
        n_conditional_channels (int, optional): Number of channels for conditional melspectrogram (default is 1).

    Attributes:
        dilated_conv (Conv1d): Dilated convolutional layer in the residual block.
        diffusion_projection (Linear): Projection layer for diffusion dimension.
        conditioner_projection (Conv2d or None): Conditional projection layer (None for unconditional model).
        output_projection (Conv1d): Output projection layer for the residual block.

    Methods:
        forward(x, diffusion_step, conditioner=None): Performs a forward pass through the residual block.

    """
    def __init__(
        self,
        n_residual_channels: int,
        dilation: int,
        kernel: int = 3,
        diffusion_dim: int = 512,
        n_mels: int = None,
        n_conditional_channels: int = 1,
    ):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            n_residual_channels,
            2 * n_residual_channels,
            kernel,
            padding='same',
            dilation=dilation
        )
        self.diffusion_projection = nn.Linear(diffusion_dim, n_residual_channels)

        if n_mels is not None: # conditional model
            self.conditioner_projection = nn.Conv2d(n_mels, 2 * n_residual_channels, (n_conditional_channels, 1))
        else: # unconditional model
            self.conditioner_projection = None

        # FIXME: 2 * n_residual_channels could be changed to 2*output_channels to make U-net-like architecture possible
        self.output_projection = nn.Conv1d(n_residual_channels, 2 * n_residual_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        diffusion_step: torch.Tensor,
        conditioner: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass through the ResidualBlock.

        Args:
            x (torch.Tensor): Input tensor to the block.
            diffusion_step (torch.Tensor): Tensor representing the diffusion step.
            conditioner (torch.Tensor, optional): Conditioning tensor for the model (default is None).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of the output and the intermediate skip connection.

        """
        if (conditioner is None) != (self.conditioner_projection is None):
            raise ValueError('Conditioner and projection should be both None or not None')

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step

        if self.conditioner_projection is None: # using a unconditional model
            y = self.dilated_conv(y)
        else:
            y = self.dilated_conv(y) + torch.squeeze(self.conditioner_projection(conditioner), 2)

        gate, filter_ = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip
