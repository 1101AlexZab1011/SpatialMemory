from dataclasses import dataclass
from utils.diffwave.layers import DiffusionEmbedding, SpectrogramUpsampler


@dataclass
class ConditionerParams:
    """
    Data class defining parameters for the conditioner model.

    Args:
        n_mels (int): Number of mel-frequency bands.
        upsampler (SpectrogramUpsampler): Spectrogram upsampler for the conditioner model (default is SpectrogramUpsampler).
        n_channels (int): Number of input channels (default is 1).
        kernel (tuple[int, int]): Size of the convolutional kernel (default is (3, 32)).
        stride (tuple[int, int]): Stride value for the convolutional operation (default is (1, 16)).
        padding (tuple[int, int]): Padding applied to the input tensor (default is (1, 8)).
        negative_slope (float): Slope value for the LeakyReLU activation (default is 0.4).
        n_layers (int): Number of upsampling layers (default is 2).

    Attributes:
        n_mels (int): Number of mel-frequency bands.
        upsampler (SpectrogramUpsampler): Spectrogram upsampler instance.
        n_channels (int): Number of input channels for the conditioner model.
        kernel (tuple[int, int]): Size of the convolutional kernel.
        stride (tuple[int, int]): Stride value for the convolutional operation.
        padding (tuple[int, int]): Padding applied to the input tensor.
        negative_slope (float): Slope value for the LeakyReLU activation.
        n_layers (int): Number of upsampling layers.

    Methods:
        __post_init__(): Initializes the upsampler instance using the provided parameters.

    """
    n_mels: int
    upsampler: SpectrogramUpsampler = SpectrogramUpsampler
    n_channels: int = 1
    kernel: tuple[int, int] = (3, 32)
    stride: tuple[int, int] = (1, 16)
    padding: tuple[int, int] = (1, 8)
    negative_slope: float = 0.4
    n_layers: int = 2

    def __post_init__(self):
        self.upsampler = self.upsampler(
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            negative_slope=self.negative_slope,
            n_layers=self.n_layers,
        )

@dataclass
class ResidualParams:
    """
    Data class defining parameters for the residual blocks.

    Args:
        n_residual_layers (int): Number of residual layers.
        n_residual_channels (int): Number of channels for the residual blocks.
        dilation_cycle_length (int): Length of the dilation cycle.
        kernel (int): Size of the convolutional kernel (default is 3).
        conditioner (ConditionerParams): Conditioner parameters (default is None).

    Attributes:
        n_residual_layers (int): Number of residual layers.
        n_residual_channels (int): Number of channels for the residual blocks.
        dilation_cycle_length (int): Length of the dilation cycle.
        kernel (int): Size of the convolutional kernel.
        conditioner (ConditionerParams): Conditioner parameters.

    """
    n_residual_layers: int
    n_residual_channels: int
    dilation_cycle_length: int
    kernel: int = 3
    conditioner: ConditionerParams = None


@dataclass
class DiffusionParams:
    """
    Data class defining parameters for the diffusion model.

    Args:
        max_steps (int): Maximum number of steps in diffusion.
        embedding_dim (int): Dimension for the embedding (default is 64).
        diffusion_dim (int): Dimension for diffusion (default is 512).
        embedding (DiffusionEmbedding): Embedding instance for diffusion.

    Attributes:
        max_steps (int): Maximum number of steps in diffusion.
        embedding_dim (int): Dimension for the embedding.
        diffusion_dim (int): Dimension for diffusion.
        embedding (DiffusionEmbedding): Embedding instance for diffusion.

    Methods:
        __post_init__(): Initializes the embedding instance using the provided parameters.

    """
    max_steps: int
    embedding_dim: int = 64
    diffusion_dim: int = 512
    embedding: DiffusionEmbedding = DiffusionEmbedding

    def __post_init__(self):
        self.embedding = self.embedding(self.max_steps, self.embedding_dim, self.diffusion_dim)