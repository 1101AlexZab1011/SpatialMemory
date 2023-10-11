import numpy as np
import os
import configparser
from abc import ABC, abstractmethod
from numba import jit
from utils.math.geometry import calculate_polar_distance
from utils.preprocessing.environment import Coordinates2D, Geometry

from utils.structures.synapses import NeuralMass, NeuralWeights

class AbstractGenerator(ABC):
    """
    An abstract base class for neural cells generators that generate data based on specified parameters.
    """

    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Generate data based on specified parameters.

        Returns:
            np.ndarray: Generated data.
        """
        pass


class GCGenerator(AbstractGenerator):
    """
    A class for generating grid cell firing rate maps based on specified parameters.
    """
    def __init__(
        self,
        res: float,
        x_max: int,
        y_max: int,
        n_mod: int,
        n_per_mod: int,
        f_mods: np.ndarray,
        FAC: np.ndarray,
        r_size: np.ndarray,
        orientations: np.ndarray,
        save_path: str
    ):
        """
        Initialize the GridCellFRGenerator.

        Args:
            res (float): Resolution of the grid.
            x_max (int): Maximum X coordinate.
            y_max (int): Maximum Y coordinate.
            n_mod (int): Number of grid modules.
            n_per_mod (int): Number of offsets per module.
            f_mods (np.ndarray): Frequencies for different modules.
            fac (np.ndarray): Scaling offset steps to cover the complete rhombus.
            r_size (np.ndarray): Template radius in number of pixels for each module.
            orientations (np.ndarray): Orientations for different modules.
            save_path (str): Path where generated data will be saved.
        """
        if not os.path.exists(save_path):
            raise OSError('The save path does not exist: {}'.format(save_path))

        self.n_mod = n_mod
        self.n_per_mod = n_per_mod
        self.res = res
        self.x_max = x_max
        self.y_max = y_max

        self.f_mods = f_mods
        self.FAC = FAC
        self.r_size = r_size
        self.orientations = orientations
        self.save_path = save_path

    def generate_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate coordinates and mesh grids.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: X coordinates, Y coordinates, and their mesh grid.
        """
        X = np.arange(0, self.x_max + self.res, self.res)
        Y = np.arange(0, self.y_max + self.res, self.res)
        Xg, Yg = np.meshgrid(X, Y)
        XY = np.column_stack((Xg.ravel(), Yg.ravel()))
        return X, Y, XY

    def get_orientation_params(self, index: int) -> tuple[float, float, np.ndarray]:
        """
        Get parameters for a specific orientation index.

        Args:
            index (int): Index of the orientation.

        Returns:
            Tuple[float, float, np.ndarray]: Frequency, factor, and rotation matrix.
        """
        F = self.f_mods[index]
        fac = self.FAC[index]
        orientation = self.orientations[index]
        R = np.array([[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
        return F, fac, R

    @staticmethod
    def get_basis_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the basis vectors for the triangular grid pattern.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Three basis vectors representing triangular directions.
        """
        b0 = np.array([np.cos(0), np.sin(0)])
        b1 = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)])
        b2 = np.array([np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)])
        return b0, b1, b2

    @staticmethod
    def get_offset_vectors(fac: float, F: float, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Get offset vectors for the triangular grid pattern.

        Args:
            fac (float): Factor for scaling offset vectors.
            F (float): Frequency parameter.
            R (np.ndarray): Rotation matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two offset vectors based on the given parameters.
        """
        x_off_base1 = 0
        y_off_base1 = fac * (1 / F)
        x_off_base2 = fac * (1 / F) * np.cos(np.pi / 6)
        y_off_base2 = fac * (1 / F) * np.sin(np.pi / 6)
        off_vec1 = np.dot(R, np.array([x_off_base1, y_off_base1]))
        off_vec2 = np.dot(R, np.array([x_off_base2, y_off_base2]))
        return off_vec1, off_vec2

    @staticmethod
    def get_offset(w: int, j: int, off_vec1: np.ndarray, off_vec2: np.ndarray) -> np.ndarray:
        """
        Calculate the offset vector for the grid cell.

        Args:
            w (int): Row index.
            j (int): Column index.
            off_vec1 (np.ndarray): First offset vector.
            off_vec2 (np.ndarray): Second offset vector.

        Returns:
            np.ndarray: Computed offset vector for the grid cell.
        """
        off = (j - 1) / 10 * off_vec1 + (w - 1) / 10 * off_vec2
        return off

    @staticmethod
    def get_z_values(
        F: float,
        XY: np.ndarray,
        R: np.ndarray,
        Off: np.ndarray,
        b0: np.ndarray,
        b1: np.ndarray,
        b2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the z values for grid cell positions.

        Args:
            F (float): Frequency parameter.
            XY (np.ndarray): Array of XY coordinates.
            R (np.ndarray): Rotation matrix.
            Off (np.ndarray): Offset vector.
            b0 (np.ndarray): First basis vector.
            b1 (np.ndarray): Second basis vector.
            b2 (np.ndarray): Third basis vector.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Calculated z values for the three basis vectors.
        """
        z0 = np.sum((np.tile(np.dot(R, b0), (XY.shape[0], 1)) * (F * XY + np.tile(Off, (XY.shape[0], 1)))), axis=1)
        z1 = np.sum((np.tile(np.dot(R, b1), (XY.shape[0], 1)) * (F * XY + np.tile(Off, (XY.shape[0], 1)))), axis=1)
        z2 = np.sum((np.tile(np.dot(R, b2), (XY.shape[0], 1)) * (F * XY + np.tile(Off, (XY.shape[0], 1)))), axis=1)
        return z0, z1, z2

    @staticmethod
    def get_FR_map(z0: np.ndarray, z1: np.ndarray, z2: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """
        Calculate the firing rate map based on the given z values.

        Args:
            z0 (np.ndarray): Z values for basis vector 0.
            z1 (np.ndarray): Z values for basis vector 1.
            z2 (np.ndarray): Z values for basis vector 2.
            shape (Tuple[int, int]): Shape of the firing rate map.

        Returns:
            np.ndarray: Firing rate map with values computed from the z values.
        """
        FRmap = np.cos(z0) + np.cos(z1) + np.cos(z2)
        FRmap = FRmap / np.max(FRmap)
        FRmap[FRmap < 0] = 0
        return np.reshape(FRmap, shape)

    @staticmethod
    def normalize(matrix: np.ndarray) -> np.ndarray:
        """
        Normalize a matrix by dividing all its elements by the maximum value.

        Args:
            matrix (np.ndarray): Matrix to be normalized.

        Returns:
            np.ndarray: Normalized matrix.
        """
        matrix = matrix / np.max(matrix)
        return matrix

    def populate(
        self,
        FRmap: np.ndarray,
        GC_FR_maps: np.ndarray,
        GC_FR_maps_SD: np.ndarray,
        w: int,
        j: int,
        i: int
    ) -> None:
        """
        Populate the grid cell firing rate maps with values computed from the given firing rate map.

        Args:
            FRmap (np.ndarray): Firing rate map to be populated.
            GC_FR_maps (np.ndarray): Grid cell firing rate maps.
            GC_FR_maps_SD (np.ndarray): Standard deviations for grid cell firing rate maps.
            w (int): Row index.
            j (int): Column index.
            i (int): Orientation index.

        Returns:
            None
        """
        shape = FRmap.shape
        tmp = np.zeros((int(shape[0] / 10), int(shape[1] / 10)))
        for k in range(int(shape[0] / 10)):
            for l in range(int(shape[1] / 10)):
                tmp[k, l] = np.mean(
                    FRmap[
                        5 + (k - 1) * 10 - 4:5 + (k - 1) * 10 + 4,
                        5 + (l - 1) * 10 - 4:5 + (l - 1) * 10 + 4
                    ]
                )

        GC_FR_maps[:, :, (j - 1) * int(np.sqrt(self.n_per_mod)) + w, i] = FRmap
        GC_FR_maps_SD[:, :, (j - 1) * int(np.sqrt(self.n_per_mod)) + w, i] = tmp / (np.max(tmp) + 1e-7)

    def save(self, GC_FR_maps: np.ndarray, GC_FR_maps_SD: np.ndarray) -> None:
        """
        Save grid cell firing rate maps and associated maps with standard deviations to files.

        Args:
            GC_FR_maps (np.ndarray): Grid cell firing rate maps.
            GC_FR_maps_SD (np.ndarray): Grid cell firing rate maps with standard deviations.

        Returns:
            None
        """
        np.save(
            os.path.join(self.save_path, 'GC_FR_maps_BB.npy'),
            GC_FR_maps
        )
        np.save(
            os.path.join(self.save_path, 'GC_FR_maps_SD.npy'),
            GC_FR_maps_SD
        )

    def generate(self, save: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate grid cell firing rate maps.

        Args:
            save (bool, optional): Whether to save the generated maps. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Generated firing rate maps and associated maps with standard deviations.
        """
        X, Y, XY = self.generate_coordinates()
        shape = (len(X), len(Y))
        b0, b1, b2 = self.get_basis_vectors()

        GC_FR_maps = np.zeros((*shape, self.n_per_mod, self.n_mod))
        GC_FR_maps_SD = np.zeros((shape[0] // 10, shape[1] // 10, self.n_per_mod, self.n_mod))

        for i in range(len(self.orientations)):
            F, fac, R = self.get_orientation_params(i)
            off_vec1, off_vec2 = self.get_offset_vectors(fac, F, R)

            for w in range(int(np.sqrt(self.n_per_mod))):
                for j in range(int(np.sqrt(self.n_per_mod))):
                    Off = self.get_offset(w, j, off_vec1, off_vec2)
                    z0, z1, z2 = self.get_z_values(F, XY, R, Off, b0, b1, b2)
                    FRmap = self.normalize(self.get_FR_map(z0, z1, z2, shape))

                    self.populate(
                        FRmap,
                        GC_FR_maps,
                        GC_FR_maps_SD,
                        w, j, i
                    )

        if save:
            self.save(GC_FR_maps, GC_FR_maps_SD)

        return GC_FR_maps, GC_FR_maps_SD


@jit(nopython=True, parallel=True)
def calculate_gc2pc_weights(gc_fr_maps, res, n_mod, n_per_mod, n_pc_mod, x_max, y_max):
    n_GCs = n_mod * n_per_mod
    n_PCs = n_pc_mod**2
    GC2PCwts = np.zeros((n_PCs, n_GCs))
    shape = gc_fr_maps.shape
    PCtmplte = np.zeros(shape)

    for x in np.arange(res, x_max + res, res) * 2:
        for y in np.arange(res, y_max + res, res) * 2:
            PC = PCtmplte.copy()
            PC[int(x), int(y)] = 1
            for i in range(n_mod):
                for j in range(n_per_mod):
                    # fire_map = np.max(np.multiply(PC, np.expand_dims(gc_fr_maps[:, :, j, i], (2, 3))))
                    fr_maps = gc_fr_maps[:, :, j, i][:, :, np.newaxis, np.newaxis]
                    print(PC.shape, fr_maps.shape, np.repeat(fr_maps, [1, PC.shape[-1]] ).shape)#, np.repeat(fr_maps, [1, 1, PC.shape[1], PC.shape[2]] ).shape)
                    # fire_map = np.max(np.multiply(PC, gc_fr_maps[:, :, j, i][:, :, np.newaxis, np.newaxis]))
                    fire_map = np.max(np.multiply(PC, np.repeat(gc_fr_maps[:, :, j, i][:, :, np.newaxis, np.newaxis], [1, 1, PC.shape[1], PC.shape[2]] )))
                    GC2PCwts[int((x - 1) * n_pc_mod + y), int((i - 1) * n_per_mod + j)] = fire_map

    return GC2PCwts


class PCGenerator(AbstractGenerator):
    """
    A class for calculating place cell weight matrices based on grid cell firing rate maps and specified parameters.
    """
    def __init__(
        self,
        res: float,
        x_max: int,
        y_max: int,
        n_mod: int,
        n_per_mod: int,
        n_pc_mod: int,
        gc_fr_maps_path: str,
        save_path: str
    ):
        """
        Initialize the PlaceCellWeightCalculator.

        Args:
            res (float): Resolution of the grid.
            x_max (int): Maximum X coordinate.
            y_max (int): Maximum Y coordinate.
            n_mod (int): Number of grid modules.
            n_per_mod (int): Number of offsets per module.
            n_pc_mod (int): Number of place cell modules.
            gc_fr_maps_path (str): Path to the grid cell firing rate maps file.
            save_path (str): Path where generated data will be saved.
        """

        if not os.path.exists(gc_fr_maps_path):
            raise OSError('The grid cell firing rate maps file does not exist: {}'.format(gc_fr_maps_path))
        if not os.path.exists(save_path):
            raise OSError('The save path does not exist: {}'.format(save_path))

        self.gc_fr_maps = np.load(gc_fr_maps_path)
        self.config = configparser.ConfigParser()
        self.config.read(ini_path)

        self.res = res
        self.x_max = x_max
        self.y_max = y_max
        self.n_mod = n_mod
        self.n_per_mod = n_per_mod
        self.n_pc_mod = n_pc_mod
        self.n_GCs = self.n_mod * self.n_per_mod
        self.n_PCs = self.n_pc_mod**2

        self.save_path = save_path

    def save(self, GC2PCwts: np.ndarray) -> None:
        """
        Save the calculated place cell weight matrix.

        Args:
            GC2PCwts (np.ndarray): Place cell weight matrix.
        """
        np.save(
            os.path.join(self.save_path, 'GC2PCwts_BB.npy'),
            GC2PCwts
        )


    def generate(self, save: bool = False) -> np.ndarray:
        """
        Generate place cell weight matrix based on grid cell firing rate maps.

        Args:
            save (bool, optional): Whether to save the generated weight matrix. Defaults to False.

        Returns:
            np.ndarray: Generated place cell weight matrix.
        """
        GC2PCwts = calculate_gc2pc_weights(self.gc_fr_maps, self.res, self.n_mod, self.n_per_mod, self.n_pc_mod, self.x_max, self.y_max)

        if save:
            self.save(GC2PCwts)

        return GC2PCwts


class MTLGenerator(AbstractGenerator):
    """
    MTLGenerator represents a generator for Medial Temporal Lobe (MTL) neural network weights.
    It calculates and initializes weights for connections between various components of the network.

    Args:
        res (float): Resolution for spatial discretization.
        x_max (int): Maximum x-coordinate for spatial grid.
        y_max (int): Maximum y-coordinate for spatial grid.
        r_max (int): Maximum radial distance for polar grid.
        x_min (int): Minimum x-coordinate for spatial grid.
        y_min (int): Minimum y-coordinate for spatial grid.
        h_sig (float): Spatial spread parameter for activation functions.
        polar_dist_res (int): Resolution for polar distance grid.
        polar_ang_res (int): Resolution for polar angle grid.
        geometry (Geometry): Geometry object representing spatial parameters.

    Attributes:
        res (float): Resolution for spatial discretization.
        x_max (int): Maximum x-coordinate for spatial grid.
        y_max (int): Maximum y-coordinate for spatial grid.
        r_max (int): Maximum radial distance for polar grid.
        x_min (int): Minimum x-coordinate for spatial grid.
        y_min (int): Minimum y-coordinate for spatial grid.
        h_sig (float): Spatial spread parameter for activation functions.
        polar_dist_res (int): Resolution for polar distance grid.
        polar_ang_res (int): Resolution for polar angle grid.
        geometry (Geometry): Geometry object representing spatial parameters.
        sigma_th (float): Standard deviation for orientation tuning.
        sigma_r0 (float): Standard deviation for polar grid tuning.
        alpha_small (float): Small positive value to avoid division by zero.

    Methods:
        get_coords() -> Tuple[Coordinates2D, int, Coordinates2D]:
            Calculate the spatial coordinates and dimensions of the neural network grid.

        get_bvc_params() -> Tuple[int, np.ndarray, np.ndarray]:
            Calculate parameters for Boundary Vector Cell (BVC) connections.

        get_perifirical_cells_params() -> Tuple[int, np.ndarray]:
            Calculate parameters for Perirhinal Cells (PR).

        get_h_sq_distances(coords: Coordinates2D, n_neurons_total: int) -> np.ndarray:
            Calculate squared distances between hidden neurons.

        initialize_h2h_weights(h_sq_distances: np.ndarray, h_sig: float) -> np.ndarray:
            Initialize weights for hidden-to-hidden connections.

        initialize_pr2pr_weights(n_pr: int) -> np.ndarray:
            Initialize weights for PR-to-PR connections.

        initialize_bvc2bvc_weights(n_bvc: int) -> np.ndarray:
            Initialize weights for BVC-to-BVC connections.

        initialize_auto_weights(h_sq_distances: np.ndarray, h_sig: float, n_pr: int, n_bvc: int):
            Initialize auto-weights for hidden, perirhinal, and BVC layers.

        initialize_cross_weights(
            n_h_neurons_total: int,
            n_bvc: int,
            n_pr: int,
            coords: Coordinates2D,
            bvc_ang: np.ndarray,
            bvc_dist: np.ndarray,
            p_reactivations: np.ndarray
        ) -> Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]:
            Initialize cross-weights between hidden, perirhinal, and BVC layers.

        invert_weights(*weights: np.ndarray) -> Tuple[np.ndarray, ...]:
            Invert weight matrices.

        normalize_weights(
            bvc2h_weights: np.ndarray,
            h2bvc_weights: np.ndarray,
            bvc2pr_weights: np.ndarray,
            pr2bvc_weights: np.ndarray,
            h2pr_weights: np.ndarray,
            pr2h_weights: np.ndarray
        ) -> Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray
        ]:
            Normalize weight matrices.

        generate() -> NeuralMass:
            Generate neural model weights for Multi-Task Learning (MTL) model.

    Example:

        # Initialize MTLGenerator with configuration parameters
        mtl_generator = MTLGenerator(
            res,
            x_max,
            y_max,
            r_max,
            x_min,
            y_min,
            h_sig,
            polar_dist_res,
            polar_ang_res,
            geometry
        )

        # Generate neural network weights
        weights = mtl_generator.generate()

    """
    def __init__(
        self,
        res: float,
        x_max: int,
        y_max: int,
        r_max: int,
        x_min: int,
        y_min: int,
        h_sig: float,
        polar_dist_res: int,
        polar_ang_res: int,
        geometry: Geometry
    ):
        """
        Initialize MTLGenerator with the specified parameters.

        Args:
            res (float): Resolution for spatial discretization.
            x_max (int): Maximum x-coordinate for spatial grid.
            y_max (int): Maximum y-coordinate for spatial grid.
            r_max (int): Maximum radial distance for polar grid.
            x_min (int): Minimum x-coordinate for spatial grid.
            y_min (int): Minimum y-coordinate for spatial grid.
            h_sig (float): Spatial spread parameter for activation functions.
            polar_dist_res (int): Resolution for polar distance grid.
            polar_ang_res (int): Resolution for polar angle grid.
            geometry (Geometry): Geometry object representing spatial parameters.
        """
        self.res = res
        self.x_max = x_max
        self.y_max = y_max
        self.r_max = r_max
        self.x_min = x_min
        self.y_min = y_min
        self.h_sig = h_sig
        self.polar_dist_res = polar_dist_res
        self.polar_ang_res = polar_ang_res
        self.geometry = geometry

        self.sigma_th = np.sqrt(0.05)
        self.sigma_r0 = 0.08
        self.alpha_small = 1e-6

    def get_coords(self) -> tuple[Coordinates2D, int, Coordinates2D]:
        """
        Calculate spatial coordinates and dimensions of the neural network grid.

        Returns:
            Tuple[Coordinates2D, int, Coordinates2D]: A tuple containing spatial coordinates, total number of neurons,
            and the dimensions of the grid.
        """
        n_neurons = Coordinates2D( #  Total H neurons in each dir
            int((self.geometry.params.max_train.x - self.geometry.params.min_train.x)/res),
            int((self.geometry.params.max_train.y - self.geometry.params.min_train.y)/res),
        )
        n_neurons_total = n_neurons.x * n_neurons.y #  Total H neurons
        coords = Coordinates2D(*np.meshgrid( # x,y cords for all H neurons
            np.arange(
                self.geometry.params.min_train.x + self.res/2,
                self.geometry.params.min_train.x + (n_neurons.x - 0.5) * self.res + self.res,
                self.res
            ),
            np.arange(
                self.geometry.params.min_train.y + self.res/2,
                self.geometry.params.min_train.y + (n_neurons.y - 0.5) * self.res + self.res,
                self.res
            )
        ))
        return coords, n_neurons_total, n_neurons

    def get_bvc_params(self) -> tuple[int, np.ndarray, np.ndarray]:
        """
        Calculate parameters for Boundary Vector Cell (BVC) connections.

        Returns:
            Tuple[int, np.ndarray, np.ndarray]: A tuple containing the number of BVCs, BVC distances, and BVC angles.
        """
        n_bvc_r = self.r_max // self.polar_dist_res # Num BVCs along a radius
        n_bvc_theta = int(np.floor( (2*np.pi - 0.01) / self.polar_ang_res ) + 1) # Num BVCs in a ring
        n_bvc = n_bvc_r * n_bvc_theta
        polar_dist = calculate_polar_distance(self.r_max)

        polar_ang = np.arange(0, n_bvc_theta * self.polar_ang_res, self.polar_ang_res)
        p_dist, p_ang = np.meshgrid(polar_dist, polar_ang) #  polar coords of all BVC neurons

        bvc_dist = p_dist.flatten() # Same, but in column vector
        bvc_ang = p_ang.flatten()

        bvc_ang = bvc_ang - 2 * np.pi * (bvc_ang > np.pi) # Make sure angles in correct range

        return n_bvc, bvc_dist, bvc_ang

    def get_perifirical_cells_params(self) -> tuple[int, np.ndarray]:
        """
        Calculate parameters for perirhinal cells.

        Returns:
            Tuple[int, np.ndarray]: A tuple containing the number of perirhinal cells and perirhinal reactivations.
        """
        n_pr = self.geometry.n_textures # One perirhinal neuron for each identity/texture
        p_reactivations = np.eye(n_pr) # identity matrix
        return n_pr, p_reactivations

    @staticmethod
    def get_h_sq_distances(coords: Coordinates2D, n_neurons_total: int) -> np.ndarray:
        """
        Calculate squared distances between hidden neurons.

        Args:
            coords (Coordinates2D): Spatial coordinates of neurons.
            n_neurons_total (int): Total number of neurons.

        Returns:
            np.ndarray: Array of squared distances between neurons.
        """
        h_separations = Coordinates2D(
            (np.outer(coords.x, np.ones(n_neurons_total)) - np.outer(coords.x, np.ones(n_neurons_total)).T).T,
            (np.outer(coords.y, np.ones(n_neurons_total)) - np.outer(coords.y, np.ones(n_neurons_total)).T).T
        )

        # Calculate square distances
        h_sq_distances = h_separations.x**2 + h_separations.y**2

        return h_sq_distances

    @staticmethod
    def initialize_h2h_weights(h_sq_distances: np.ndarray, h_sig: float) -> np.ndarray:
        """
        Initialize weights for hidden-to-hidden connections.

        Args:
            h_sq_distances (np.ndarray): Squared distances between hidden neurons.
            h_sig (float): Spatial spread parameter for activation functions.

        Returns:
            np.ndarray: Initialized weights for hidden-to-hidden connections.
        """
        h2h_weights = np.exp(-h_sq_distances / (h_sig**2))
        return h2h_weights

    @staticmethod
    def initialize_pr2pr_weights(n_pr: int) -> np.ndarray:
        """
        Initialize weights for perirhinal-to-perirhinal connections.

        Args:
            n_pr (int): Number of perirhinal neurons.

        Returns:
            np.ndarray: Initialized weights for perirhinal-to-perirhinal connections.
        """
        # Initialize pr2pr_weights
        return np.zeros((n_pr, n_pr))

    @staticmethod
    def initialize_bvc2bvc_weights(n_bvc: int) -> np.ndarray:
        """
        Initialize weights for BVC-to-BVC connections.

        Args:
            n_bvc (int): Number of Boundary Vector Cells (BVCs).

        Returns:
            np.ndarray: Initialized weights for BVC-to-BVC connections.
        """
        # Initialize bvc2bvc_weights
        return np.zeros((n_bvc, n_bvc))

    def initialize_auto_weights(
        self,
        h_sq_distances: np.ndarray,
        h_sig: float,
        n_pr: int,
        n_bvc: int
    ):
        """
        Initialize auto-weights for hidden, perirhinal, and BVC layers.

        Args:
            h_sq_distances (np.ndarray): Squared distances between hidden neurons.
            h_sig (float): Spatial spread parameter for activation functions.
            n_pr (int): Number of perirhinal neurons.
            n_bvc (int): Number of Boundary Vector Cells (BVCs).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Initialized weights for auto-connections.
        """
        return self.initialize_h2h_weights(h_sq_distances, h_sig), self.initialize_pr2pr_weights(n_pr), self.initialize_bvc2bvc_weights(n_bvc)

    def initialize_cross_weights(
        self,
        n_h_neurons_total: int,
        n_bvc: int,
        n_pr: int,
        coords: Coordinates2D,
        bvc_ang: np.array,
        bvc_dist: np.ndarray,
        p_reactivations: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Initialize cross-weights between hidden, perirhinal, and BVC layers.

        Args:
            n_h_neurons_total (int): Total number of hidden neurons.
            n_bvc (int): Number of Boundary Vector Cells (BVCs).
            n_pr (int): Number of perirhinal neurons.
            coords (Coordinates2D): Spatial coordinates of neurons.
            bvc_ang (np.ndarray): BVC angles.
            bvc_dist (np.ndarray): BVC distances.
            p_reactivations (np.ndarray): Perirhinal reactivations.

        Returns:
            Tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]: Initialized cross-weights between layers.
        """
        bvc2h_weights = np.zeros((n_h_neurons_total, n_bvc))
        bvc2pr_weights = np.zeros((n_pr, n_bvc))
        pr2h_weights = np.zeros((n_h_neurons_total, n_pr))
        h2pr_weights = pr2h_weights.T

        for location in range(self.geometry.visible_plane.training_locations.shape[0]):
            pos_x = self.geometry.visible_plane.training_locations[location, 0]
            pos_y = self.geometry.visible_plane.training_locations[location, 1]

            non_nan_indices = np.where(~np.isnan(self.geometry.visible_plane.coords.x[location, :]))[0]
            visible_boundary_points = Coordinates2D(
                self.geometry.visible_plane.coords.x[location, non_nan_indices] - pos_x,
                self.geometry.visible_plane.coords.y[location, non_nan_indices] - pos_y
            )
            boundary_point_texture = self.geometry.visible_plane.textures[location, non_nan_indices]

            boundary_theta, boundary_r = np.arctan2(visible_boundary_points.y, visible_boundary_points.x), np.sqrt(visible_boundary_points.x**2 + visible_boundary_points.y**2)
            boundary_r[boundary_r < self.polar_dist_res] = self.polar_dist_res

            h_activarions = np.exp(-((coords.x.reshape((-1, 1)) - pos_x)**2 + (coords.y.reshape((-1, 1)) - pos_y)**2) / (self.h_sig**2))
            bvc_activations = np.zeros(n_bvc)
            bvc2pr_weights_contrib = np.zeros(bvc2pr_weights.shape)
            h2pr_weights_contrib = np.zeros(h2pr_weights.shape)

            for boundary_point in range(visible_boundary_points.x.size):
                angle_acute = np.abs(bvc_ang - boundary_theta[boundary_point])
                angle_obtuse = 2 * np.pi - np.abs(-bvc_ang + boundary_theta[boundary_point])
                angle_difference = (angle_acute < np.pi)*angle_acute + (angle_acute > np.pi)*angle_obtuse
                sigma_r = (boundary_r[boundary_point] + 8) * self.sigma_r0
                delayed_bvc_activations = (
                    (1 / boundary_r[boundary_point])
                    * (np.exp(-(angle_difference / self.sigma_th)**2)
                    * np.exp(-((bvc_dist - boundary_r[boundary_point]) / sigma_r)**2))
                    * (bvc_activations <= 1)
                )
                bvc_activations += delayed_bvc_activations
                bvc2pr_weights_contrib += np.outer(p_reactivations[:, int(boundary_point_texture[boundary_point]) - 1], delayed_bvc_activations)
                h2pr_weights_contrib += np.outer(p_reactivations[:, int(boundary_point_texture[boundary_point]) - 1], h_activarions)

            bvc2h_weights_contrib = np.outer(h_activarions, bvc_activations)

            bvc2h_weights += bvc2h_weights_contrib
            bvc2pr_weights += bvc2pr_weights_contrib
            h2pr_weights += h2pr_weights_contrib

        h2bvc_weights, pr2bvc_weights, pr2h_weights = self.invert_weights(bvc2h_weights, bvc2pr_weights, h2pr_weights)

        # Post-synaptic normalization
        bvc2h_weights, h2bvc_weights, bvc2pr_weights, pr2bvc_weights, h2pr_weights, pr2h_weights = self.normalize_weights(
            bvc2h_weights,
            h2bvc_weights,
            bvc2pr_weights,
            pr2bvc_weights,
            h2pr_weights,
            pr2h_weights
        )

        return bvc2h_weights, bvc2pr_weights, pr2h_weights, h2pr_weights, h2bvc_weights, pr2bvc_weights

    def invert_weights(self, *weights: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Invert weight matrices.

        Args:
            *weights (np.ndarray): Variable number of weight matrices to be inverted.

        Returns:
            Tuple[np.ndarray, ...]: Tuple of inverted weight matrices.
        """
        return tuple([weight.T for weight in weights])

    def normalize_weights(
        self,
        bvc2h_weights: np.ndarray,
        h2bvc_weights: np.ndarray,
        bvc2pr_weights: np.ndarray,
        pr2bvc_weights: np.ndarray,
        h2pr_weights: np.ndarray,
        pr2h_weights: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray
    ]:
        """
        Normalize weight matrices.

        Args:
            bvc2h_weights (np.ndarray): BVC-to-hidden weights.
            h2bvc_weights (np.ndarray): Hidden-to-BVC weights.
            bvc2pr_weights (np.ndarray): BVC-to-perirhinal weights.
            pr2bvc_weights (np.ndarray): Perirhinal-to-BVC weights.
            h2pr_weights (np.ndarray): Hidden-to-perirhinal weights.
            pr2h_weights (np.ndarray): Perirhinal-to-hidden weights.

        Returns:
            Tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray
            ]: Normalized weight matrices.
        """
        bvc2h_weights = bvc2h_weights / (np.sum(bvc2h_weights, axis=1, keepdims=True))
        h2bvc_weights = h2bvc_weights / (np.sum(h2bvc_weights, axis=1, keepdims=True))

        bvc2pr_weights = bvc2pr_weights / ((np.sum(bvc2pr_weights, axis=1, keepdims=True) + self.alpha_small))
        pr2bvc_weights = pr2bvc_weights / (np.sum(pr2bvc_weights, axis=1, keepdims=True))
        h2pr_weights = h2pr_weights / ((np.sum(h2pr_weights, axis=1, keepdims=True) + self.alpha_small))
        pr2h_weights = pr2h_weights / (np.sum(pr2h_weights, axis=1, keepdims=True))

        return bvc2h_weights, h2bvc_weights, bvc2pr_weights, pr2bvc_weights, h2pr_weights, pr2h_weights

    def generate(self):
        """
        Generate neural model weights for Medial Temporal Lobe (MTL) model.

        Returns:
            NeuralMass: An instance of the NeuralMass class representing the generated weights.
        """
        coords, n_neurons_total, n_neurons = self.get_coords()
        n_bvc, bvc_dist, bvc_ang = self.get_bvc_params()
        n_pr, p_reactivations = self.get_perifirical_cells_params()
        h_sq_distances = self.get_h_sq_distances(coords, n_neurons_total)
        h2h_weights, pr2pr_weights, bvc2bvc_weights = self.initialize_auto_weights(h_sq_distances, self.h_sig, n_pr, n_bvc)
        bvc2h_weights, bvc2pr_weights, pr2h_weights, h2pr_weights, h2bvc_weights, pr2bvc_weights = self.initialize_cross_weights(
            n_neurons_total,
            n_bvc,
            n_pr,
            coords,
            bvc_ang,
            bvc_dist,
            p_reactivations
        )
        weights = NeuralMass(
            NeuralWeights(
                from_ = 'h',
                to = 'h',
                weights = h2h_weights
            ),
            NeuralWeights(
                from_ = 'h',
                to = 'pr',
                weights = h2pr_weights
            ),
            NeuralWeights(
                from_ = 'h',
                to = 'bvc',
                weights = h2bvc_weights
            ),
            NeuralWeights(
                from_ = 'pr',
                to = 'h',
                weights = pr2h_weights
            ),
            NeuralWeights(
                from_ = 'pr',
                to = 'pr',
                weights = pr2pr_weights
            ),
            NeuralWeights(
                from_ = 'pr',
                to = 'bvc',
                weights = pr2bvc_weights
            ),
            NeuralWeights(
                from_ = 'bvc',
                to = 'h',
                weights = bvc2h_weights
            ),
            NeuralWeights(
                from_ = 'bvc',
                to = 'pr',
                weights = bvc2pr_weights
            ),
            NeuralWeights(
                from_ = 'bvc',
                to = 'bvc',
                weights = bvc2bvc_weights
            ),
        )

        return weights


class HDGenerator(AbstractGenerator):
    """
    HDGenerator represents a generator for Head-Direction (HD) neural model weights.
    It calculates and initializes weights for connections within HD neurons and rotation neurons.

    Args:
        n_neurons (int): Number of HD neurons.
        max_amplitude (float): Maximum amplitude for weight initialization.
        sig (float): Neuron number measure as opposed to radian measure.
        n_steps (int): Number of steps for weight initialization.
        dt (float): Time step for weight initialization.
        log_size (int): Size of the log for weight initialization.
        decay (float): Decay parameter for weight initialization.

    Attributes:
        n_neurons (int): Number of HD neurons.
        max_amplitude (float): Maximum amplitude for weight initialization.
        sig (float): Neuron number measure as opposed to radian measure.
        n_steps (int): Number of steps for weight initialization.
        dt (float): Time step for weight initialization.
        log_size (int): Size of the log for weight initialization.
        decay (float): Decay parameter for weight initialization.

    Methods:
        initialize_hd2hd_weights() -> np.ndarray:
            Initialize weights for HD-to-HD connections.

        initialize_rotation_weights() -> np.ndarray:
            Initialize weights for rotation neurons.

        generate() -> NeuralMass:
            Generate neural network weights for HD and rotation neurons.

    Example:
        # Initialize HDGenerator with configuration parameters
        hd_generator = HDGenerator(
            n_neurons,
            max_amplitude,
            sig,
            n_steps,
            dt,
            log_size,
            decay
        )

        # Generate neural network weights
        weights = hd_generator.generate()
    """
    def __init__(
        self,
        n_neurons: int,
        max_amplitude: float,
        sig: float,
        n_steps: int,
        dt: float,
        log_size: int,
        decay: float,
    ):
        """
        Initialize HDGenerator with the specified parameters.

        Args:
            n_neurons (int): Number of HD neurons.
            max_amplitude (float): Maximum amplitude for weight initialization.
            sig (float): Neuron number measure as opposed to radian measure.
            n_steps (int): Number of steps for weight initialization.
            dt (float): Time step for weight initialization.
            log_size (int): Size of the log for weight initialization.
            decay (float): Decay parameter for weight initialization.
        """
        self.n_neurons = n_neurons
        self.max_amplitude = max_amplitude
        self.sig = sig
        self.n_steps = n_steps
        self.dt = dt
        self.log_size = log_size
        self.decay = decay

    def initialize_hd2hd_weights(self):
        """
        Initialize weights for HD-to-HD connections.

        Returns:
            np.ndarray: Initialized weights for HD-to-HD connections.
        """
        hd2hd_weights = np.zeros((self.n_neurons, self.n_neurons))

        x = np.arange(1, self.n_neurons + 1)
        wide_x = np.zeros((3 * self.n_neurons,))
        wide_x[:self.n_neurons] = x - self.n_neurons
        wide_x[self.n_neurons:2*self.n_neurons] = x
        wide_x[2*self.n_neurons:] = x + self.n_neurons

        for x0 in range(1, self.n_neurons + 1):
            gaussian = self.max_amplitude * (
                np.exp(-((wide_x - x0) / self.sig)**2)
                + np.exp(-((wide_x - x0 - self.n_neurons) / self.sig)**2)
                + np.exp(-((wide_x - x0 + self.n_neurons) / self.sig)**2)
            )
            hd2hd_weights[:, x0 - 1] = gaussian[self.n_neurons:2 * self.n_neurons]

        return hd2hd_weights

    def initialize_rotation_weights(self):
        """
        Initialize weights for rotation neurons.

        Returns:
            np.ndarray: Initialized weights for rotation neurons.
        """
        record = np.zeros((self.n_neurons, self.log_size))

        rotation_weights = np.zeros((self.n_neurons, self.n_neurons))
        x = np.arange(1, self.n_neurons + 1)
        wide_x = np.zeros((3 * self.n_neurons,))
        wide_x[:self.n_neurons] = x - self.n_neurons
        wide_x[self.n_neurons:2*self.n_neurons] = x
        wide_x[2*self.n_neurons:] = x + self.n_neurons

        rec_ind = 0
        for step in range(1, self.n_steps + 1):
            x0 = 2 * np.pi * np.random.rand()
            vel = 1 * np.random.rand() + 0.5

            for time in np.arange(0, 2 * np.pi / np.abs(vel), self.dt):
                xt = x0 + time * vel
                xt = xt - 2 * np.pi * (xt > 2 * np.pi)
                xt = self.n_neurons * xt / (2 * np.pi)

                Gaussian = (
                    np.exp(-((wide_x - xt) / self.sig)**2)
                    + np.exp(-((wide_x - xt - self.n_neurons) / self.sig)**2)
                    + np.exp(-((wide_x - xt + self.n_neurons) / self.sig)**2)
                )
                record *= (1 - self.dt / self.decay)
                current_activation = Gaussian[self.n_neurons:2 * self.n_neurons]
                rotation_weights += np.outer(np.sum(record, axis=1), current_activation)

                record[:, rec_ind] = current_activation
                rec_ind = (rec_ind + 1) % 100

            if step % 20 == 0:
                rotation_weights /= np.outer(np.max(rotation_weights, axis=1), np.ones(self.n_neurons))

        rotation_weights /= np.outer(np.max(rotation_weights, axis=1), np.ones(self.n_neurons))
        return rotation_weights.T

    def generate(self) -> NeuralMass:
        """
        Generate neural network weights for HD and rotation neurons.

        Returns:
            NeuralMass: An instance of the NeuralMass class representing the generated weights.
        """
        hd2hd_weights = self.initialize_hd2hd_weights()
        rotation_weights = self.initialize_rotation_weights()
        return NeuralMass(
            NeuralWeights(
                from_='hd',
                to='hd',
                weights=hd2hd_weights,
            ),
            NeuralWeights(
                from_='rot',
                to='rot',
                weights=rotation_weights,
            )
        )


if __name__ == '__main__':
    ini_path = os.path.join('.', 'cfg', 'grid_cells.ini')
    config = configparser.ConfigParser()
    config.read(ini_path)

    if not os.path.exists(ini_path):
        raise FileNotFoundError('The ini file does not exist: {}'.format(ini_path))

    n_mod = config.getint('Space', 'n_mod')
    n_per_mod = config.getint('Space', 'n_per_mod')
    res = config.getint('Space', 'res')
    x_max = config.getint('Space', 'x_max')
    y_max = config.getint('Space', 'y_max')

    f_mods = np.array(config.get('Frequencies', 'f_mods').split(','), dtype=float)
    FAC = np.array(config.get('Offsets', 'FAC').split(','), dtype=float)
    r_size = np.array(config.get('Template', 'r_size').split(','), dtype=int)
    orientations = np.array(config.get('Orientations', 'ORIs').split(','), dtype=float)

    save_path = os.path.join('.', 'data', 'grid_cells')
    generator = GCGenerator(
        res,
        x_max, y_max,
        n_mod,
        n_per_mod,
        f_mods,
        FAC,
        r_size,
        orientations,
        save_path
    )
    # generator.generate(save=True)

    #============================

    ini_path = os.path.join('.', 'cfg', 'place_cells.ini')
    config = configparser.ConfigParser()
    config.read(ini_path)

    res = config.getfloat('Parameters', 'res')
    x_max = config.getint('Parameters', 'Xmax')
    y_max = config.getint('Parameters', 'Ymax')
    n_pc_mod = config.getint('Parameters', 'NpcMod')
    grid_cells_path = os.path.join(save_path, 'GC_FR_maps_BB.npy')

    save_path = os.path.join('.', 'data', 'place_cells')

    generator = PCGenerator(
        res, x_max, y_max, n_mod, n_per_mod, n_pc_mod,
        grid_cells_path,
        save_path
    )
    generator.generate(save=True)

