from dataclasses import dataclass
import numpy as np
import os
import configparser
from abc import ABC, abstractmethod
from numba import jit
from bbtoolkit.data import WritablePickle
from bbtoolkit.math import triple_gaussian
from bbtoolkit.math.geometry import calculate_polar_distance
from bbtoolkit.preprocessing import triple_arange
from bbtoolkit.preprocessing.environment import Coordinates2D, Geometry
from scipy.sparse import csr_matrix
from bbtoolkit.structures.synapses import TensorGroup, DirectedTensor

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


@dataclass
class GCMap(WritablePickle):
    """
    Represents a Grid Cells activity Map.

    Attributes:
    -----------
    fr : np.ndarray
        An array representing the frequency rates.
    """
    fr: np.ndarray


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
        fac: np.ndarray,
        r_size: np.ndarray,
        orientations: np.ndarray
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

        self.n_mod = n_mod
        self.n_per_mod = n_per_mod
        self.res = res
        self.x_max = x_max
        self.y_max = y_max

        self.f_mods = f_mods
        self.fac = fac
        self.r_size = r_size
        self.orientations = orientations

    def get_coordinates(self) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Generate coordinates and mesh grids.

        Returns:
            tuple[np.ndarray, tuple[int, int]]: Mesh grid of X and Y coordinates and their shape.
        """
        x = np.arange(0, self.x_max + self.res, self.res)
        y = np.arange(0, self.y_max + self.res, self.res)
        xg, yg = np.meshgrid(x, y)
        # Yg and Xg are flipped because of the way matrices are indexed in matlab
        XY = np.column_stack((yg.ravel(), xg.ravel()))
        return XY, (len(x), len(y))

    def get_orientation_params(self, index: int) -> tuple[float, float, np.ndarray]:
        """
        Get parameters for a specific orientation index.

        Args:
            index (int): Index of the orientation.

        Returns:
            Tuple[float, float, np.ndarray]: Frequency, factor, and rotation matrix.
        """
        f = self.f_mods[index]
        fac = self.fac[index]
        orientation = self.orientations[index]
        r = np.array([[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
        return f, fac, r

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
    def get_offset_vectors(fac: float, f: float, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        y_off_base1 = fac * (1 / f)
        x_off_base2 = fac * (1 / f) * np.cos(np.pi / 6)
        y_off_base2 = fac * (1 / f) * np.sin(np.pi / 6)
        off_vec1 = np.dot(r, np.array([x_off_base1, y_off_base1]))
        off_vec2 = np.dot(r, np.array([x_off_base2, y_off_base2]))
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
        off = (j) / 10 * off_vec1 + (w) / 10 * off_vec2
        return off

    @staticmethod
    def get_z_values(
        f: float,
        xy: np.ndarray,
        r: np.ndarray,
        offset: np.ndarray,
        b0: np.ndarray,
        b1: np.ndarray,
        b2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the z values for grid cell positions.

        Args:
            f (float): Frequency parameter.
            xy (np.ndarray): Array of XY coordinates.
            r (np.ndarray): Rotation matrix.
            offset (np.ndarray): Offset vector.
            b0 (np.ndarray): First basis vector.
            b1 (np.ndarray): Second basis vector.
            b2 (np.ndarray): Third basis vector.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Calculated z values for the three basis vectors.
        """
        def calculate_z(r, b, xy, f, offset):
            b_dot_xy_offset = f * xy + np.tile(offset, (xy.shape[0], 1))
            b_dot_r = np.tile(np.dot(r, b), (xy.shape[0], 1))
            return np.sum(b_dot_r * b_dot_xy_offset, axis=1)

        z0 = calculate_z(r, b0, xy, f, offset)
        z1 = calculate_z(r, b1, xy, f, offset)
        z2 = calculate_z(r, b2, xy, f, offset)

        return z0, z1, z2

    @staticmethod
    def get_fr_map(z0: np.ndarray, z1: np.ndarray, z2: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
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
        fr_map = np.cos(z0) + np.cos(z1) + np.cos(z2)
        fr_map = fr_map / np.max(fr_map)
        fr_map[fr_map < 0] = 0
        return np.reshape(fr_map, shape)

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
        fr_map: np.ndarray,
        gc_fr_maps_sd: np.ndarray,
        w: int,
        j: int,
        i: int
    ) -> None:
        """
        Populate the grid cell firing rate maps with values computed from the given firing rate map.

        Args:
            FRmap (np.ndarray): Firing rate map to be populated.
            gc_fr_maps_sd (np.ndarray): Grid cell firing rate maps.
            w (int): Row index.
            j (int): Column index.
            i (int): Orientation index.

        Returns:
            None
        """
        shape = fr_map.shape
        tmp = np.zeros((int(shape[0] / 10), int(shape[1] / 10)))
        for k in range(int(shape[0] / 10)):
            for l in range(int(shape[1] / 10)):
                x_low, y_low = k * 10, l * 10
                tmp[k, l] = np.mean(
                    fr_map[
                        # in window of size 10x10 take mean of central 8x8 pixels
                        x_low + 1: x_low + 9,
                        y_low + 1: y_low + 9
                    ]
                )

        gc_fr_maps_sd[:, :, (j) * int(np.sqrt(self.n_per_mod)) + w, i] = (tmp / (np.max(tmp) + 1e-7)).T

    def generate(self) -> GCMap:
        """
        Generate grid cell firing rate maps.

        Args:
            save (bool, optional): Whether to save the generated maps. Defaults to False.

        Returns:
            GCMap: Generated firing rate maps and associated maps with standard deviations.
        """
        xy, shape = self.get_coordinates()
        b0, b1, b2 = self.get_basis_vectors()

        # Resolution is lowered by a factor of 10
        gc_fr_maps_sd = np.zeros((shape[0] // 10, shape[1] // 10, self.n_per_mod, self.n_mod))

        for i in range(len(self.orientations)):
            f, fac, r = self.get_orientation_params(i)
            off_vec1, off_vec2 = self.get_offset_vectors(fac, f, r)

            for w in range(int(np.sqrt(self.n_per_mod))):
                for j in range(int(np.sqrt(self.n_per_mod))):
                    offset = self.get_offset(w, j, off_vec1, off_vec2)
                    z0, z1, z2 = self.get_z_values(f, xy, r, offset, b0, b1, b2)
                    FRmap = self.normalize(self.get_fr_map(z0, z1, z2, shape))

                    self.populate(
                        FRmap,
                        gc_fr_maps_sd,
                        w, j, i
                    )

        return GCMap(gc_fr_maps_sd)


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
        gc_map: GCMap
    ):
        """
        Initializes the PCGenerator.

        Args:
            res (float): Resolution for calculating place cell weight matrices.
            x_max (int): Maximum value along the x-axis.
            y_max (int): Maximum value along the y-axis.
            n_mod (int): Number of modules.
            n_per_mod (int): Number of cells per module.
            gc_map (GCMap): Instance of GCMap containing grid cell firing rate maps.
        """
        self.gc_map = gc_map

        self.res = res
        self.x_max = x_max
        self.y_max = y_max
        self.n_mod = n_mod
        self.n_per_mod = n_per_mod
        self.n_points_x = int(self.x_max / self.res)
        self.n_points_y = int(self.y_max / self.res)
        self.n_gc = self.n_mod * self.n_per_mod
        self.n_pc = self.n_points_x*self.n_points_y

    def generate_gc2pc_weights(self):
        """
        Generates weights from grid cells to place cells.

        Returns:
            np.ndarray: Generated GC to PC weights.
        """
        gc2pc_weights = np.zeros((self.n_pc, self.n_points_x * self.n_points_y))
        shape = self.gc_map.fr.shape
        pc_template = np.zeros((shape[0], shape[1]))

        for x in range(self.n_points_x):
            for y in range(self.n_points_y):
                PC = pc_template.copy()
                PC[x, y] = 1

                for i in range(self.n_mod):
                    for j in range(self.n_per_mod):
                        gc2pc_weights[x * self.n_points_x + y, i * self.n_per_mod + j] = np.max(PC * self.gc_map.fr[:, :, j, i])

        return gc2pc_weights


    def generate(self) -> np.ndarray:
        """
        Generate place cell weight matrix based on grid cell firing rate maps.

        Args:
            save (bool, optional): Whether to save the generated weight matrix. Defaults to False.

        Returns:
            np.ndarray: Generated place cell weight matrix.
        """
        gc2pc_weights = self.generate_gc2pc_weights()

        return TensorGroup(
            DirectedTensor(
                from_='gc',
                to='pc',
                weights=gc2pc_weights
            )
        )


def get_boundary_activations(
    angle: np.ndarray,
    theta: float,
    dist: np.ndarray,
    radius: float,
    sigma_r0: float = 0.08,
    sigma_th: float = np.sqrt(0.05),
    mask: np.array = None
) -> np.ndarray:
    """
    Calculate boundary activations based on angle, distance, and parameters.

    This function computes boundary activations for a set of points based on their angles relative to a given theta
    (head direction) and their distances from a central point (radius). It uses the von Mises distribution for angular
    selectivity and a Gaussian distribution for radial selectivity to determine the activations.

    Args:
        angle (np.ndarray): An array of angles, representing the angles of points relative to a reference direction.
        theta (float): The reference direction (head direction) in radians.
        dist (np.ndarray): An array of distances from the central point to the target points.
        radius (float): The radial selectivity radius that determines the spread of radial activation.
        sigma_r0 (float, optional): The standard deviation of the radial Gaussian distribution when the distance is zero.
            Defaults to 0.08.
        sigma_th (float, optional): The standard deviation of the von Mises distribution for angular selectivity.
            Defaults to the square root of 0.05.
        mask (np.array, optional): An array used to mask activations. Should have the same shape as angle and dist.
            If not provided, a mask of ones (no masking) is applied. Defaults to None.

    Returns:
        np.ndarray: An array of boundary activations for each point based on the input parameters.

    Example:
        >>> angles = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        >>> central_theta = 1.2
        >>> distances = np.array([2.0, 3.0, 1.5, 4.0, 5.0])
        >>> boundary_activations = get_boundary_activations(angles, central_theta, distances, 4.0)
    """
    if mask is None:
        mask = 1

    angle_acute = np.abs(angle - theta)
    angle_obtuse = 2 * np.pi - angle_acute
    # FIXME: if angle_acute is very close to pi, it can cause numerical instability
    angle_difference = (angle_acute < np.pi) * angle_acute + (angle_acute > np.pi) * angle_obtuse
    sigma_r = (radius + 8) * sigma_r0
    activations = (
        (1 / radius)
        * (
            np.exp(-(angle_difference / sigma_th)**2)
            * np.exp( -((dist - radius) / sigma_r)**2)
        ) * mask
    )
    return activations


class MTLGenerator(AbstractGenerator):
    """
    MTLGenerator represents a generator for Medial Temporal Lobe (MTL) neural network weights.
    It calculates and initializes weights for connections between various components of the network.

    Args:
        res (float): Resolution for spatial discretization.
        r_max (int): Maximum radial distance for polar grid.
        h_sig (float): Spatial spread parameter for activation functions.
        polar_dist_res (int): Resolution for polar distance grid.
        polar_ang_res (int): Resolution for polar angle grid.
        geometry (Geometry): Geometry object representing spatial parameters.

    Attributes:
        res (float): Resolution for spatial discretization.
        r_max (int): Maximum radial distance for polar grid.
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
            Calculate squared distances between hippocampus neurons.

        initialize_h2h_weights(h_sq_distances: np.ndarray, h_sig: float) -> np.ndarray:
            Initialize weights for hippocampus-to-hippocampus connections.

        initialize_pr2pr_weights(n_pr: int) -> np.ndarray:
            Initialize weights for PR-to-PR connections.

        initialize_bvc2bvc_weights(n_bvc: int) -> np.ndarray:
            Initialize weights for BVC-to-BVC connections.

        initialize_auto_weights(h_sq_distances: np.ndarray, h_sig: float, n_pr: int, n_bvc: int):
            Initialize auto-weights for hippocampus, perirhinal, and BVC layers.

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
            Initialize cross-weights between hippocampus, perirhinal, and BVC layers.

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

        generate() -> TensorGroup:
            Generate neural model weights for Multi-Task Learning (MTL) model.

    Example:

        # Initialize MTLGenerator with configuration parameters
        mtl_generator = MTLGenerator(
            res,
            r_max,
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
        r_max: int,
        h_sig: float,
        polar_dist_res: int,
        polar_ang_res: int,
        geometry: Geometry
    ):
        """
        Initialize MTLGenerator with the specified parameters.

        Args:
            res (float): Resolution for spatial discretization.
            r_max (int): Maximum radial distance for polar grid.
            h_sig (float): Spatial spread parameter for activation functions (Sigma hill).
            polar_dist_res (int): Resolution for polar distance grid.
            polar_ang_res (int): Resolution for polar angle grid.
            geometry (Geometry): Geometry object representing spatial parameters.
        """
        self.res = res
        self.r_max = r_max
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
            int((self.geometry.params.max_train.x - self.geometry.params.min_train.x)/self.res),
            int((self.geometry.params.max_train.y - self.geometry.params.min_train.y)/self.res),
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
        n_pr = self.geometry.params.n_textures # One perirhinal neuron for each identity/texture
        p_reactivations = np.eye(n_pr) # identity matrix
        return n_pr, p_reactivations

    @staticmethod
    def get_h_sq_distances(coords: Coordinates2D, n_neurons_total: int) -> np.ndarray:
        """
        Calculate squared distances between hippocampus neurons.

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
        Initialize weights for hippocampus-to-hippocampus connections.

        Args:
            h_sq_distances (np.ndarray): Squared distances between hippocampus neurons.
            h_sig (float): Spatial spread parameter for activation functions.

        Returns:
            np.ndarray: Initialized weights for hippocampus-to-hippocampus connections.
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
        Initialize auto-weights for hippocampus, perirhinal, and BVC layers.

        Args:
            h_sq_distances (np.ndarray): Squared distances between hippocampus neurons.
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
        Initialize cross-weights between hippocampus, perirhinal, and BVC layers.

        Args:
            n_h_neurons_total (int): Total number of hippocampus neurons.
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
                delayed_bvc_activations = get_boundary_activations(
                    bvc_ang,
                    boundary_theta[boundary_point],
                    bvc_dist,
                    boundary_r[boundary_point],
                    sigma_r0=self.sigma_r0,
                    sigma_th=self.sigma_th,
                    mask=bvc_activations <= 1
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

    @staticmethod
    def invert_weights(*weights: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Invert weight matrices.

        Args:
            *weights (np.ndarray): Weight matrices to be inverted.

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
            bvc2h_weights (np.ndarray): BVC-to-hippocampus weights.
            h2bvc_weights (np.ndarray): hippocampus-to-BVC weights.
            bvc2pr_weights (np.ndarray): BVC-to-perirhinal weights.
            pr2bvc_weights (np.ndarray): Perirhinal-to-BVC weights.
            h2pr_weights (np.ndarray): hippocampus-to-perirhinal weights.
            pr2h_weights (np.ndarray): Perirhinal-to-hippocampus weights.

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
        # FIXME: In the future can be refactored, now made to be consistent with legacy code
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
            TensorGroup: An instance of the TensorGroup class representing the generated weights.
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
        weights = TensorGroup(
            DirectedTensor(
                from_ = 'h',
                to = 'h',
                weights = h2h_weights
            ),
            DirectedTensor(
                from_ = 'h',
                to = 'pr',
                weights = h2pr_weights
            ),
            DirectedTensor(
                from_ = 'h',
                to = 'bvc',
                weights = h2bvc_weights
            ),
            DirectedTensor(
                from_ = 'pr',
                to = 'h',
                weights = pr2h_weights
            ),
            DirectedTensor(
                from_ = 'pr',
                to = 'pr',
                weights = pr2pr_weights
            ),
            DirectedTensor(
                from_ = 'pr',
                to = 'bvc',
                weights = pr2bvc_weights
            ),
            DirectedTensor(
                from_ = 'bvc',
                to = 'h',
                weights = bvc2h_weights
            ),
            DirectedTensor(
                from_ = 'bvc',
                to = 'pr',
                weights = bvc2pr_weights
            ),
            DirectedTensor(
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

        generate() -> TensorGroup:
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

        wide_x = triple_arange(1, self.n_neurons + 1)

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
        wide_x = triple_arange(1, self.n_neurons + 1)

        rec_ind = 0
        for step in range(1, self.n_steps + 1):
            x0 = 2 * np.pi * np.random.rand()
            vel = 1 * np.random.rand() + 0.5

            for time in np.arange(0, 2 * np.pi / np.abs(vel), self.dt):
                xt = x0 + time * vel
                xt = xt - 2 * np.pi * (xt > 2 * np.pi)
                xt = self.n_neurons * xt / (2 * np.pi)

                g = triple_gaussian(
                    1,
                    wide_x,
                    xt,
                    self.n_neurons,
                    self.sig
                )
                record *= (1 - self.dt / self.decay)
                current_activation = g[self.n_neurons:2 * self.n_neurons]
                rotation_weights += np.outer(np.sum(record, axis=1), current_activation)

                record[:, rec_ind] = current_activation
                rec_ind = (rec_ind + 1) % 100

            if step % 20 == 0:
                rotation_weights /= np.outer(np.max(rotation_weights, axis=1), np.ones(self.n_neurons))

        rotation_weights /= np.outer(np.max(rotation_weights, axis=1), np.ones(self.n_neurons))
        return rotation_weights.T

    def generate(self) -> TensorGroup:
        """
        Generate neural model weights for HD and rotation neurons.

        Returns:
            TensorGroup: An instance of the TensorGroup class representing the generated weights.
        """
        hd2hd_weights = self.initialize_hd2hd_weights()
        rotation_weights = self.initialize_rotation_weights()
        return TensorGroup(
            DirectedTensor(
                from_='hd',
                to='hd',
                weights=hd2hd_weights,
            ),
            DirectedTensor(
                from_='rot',
                to='rot',
                weights=rotation_weights,
            )
        )


class TCGenerator(AbstractGenerator):
    """
    TCGenerator is a class that generates neural connectivity for a Transformation Circuit (TC) in a neural model.
    Head-direction provides the gain-modulation in the transformation circuit, producing directionally modulated boundary vector cells
    which connect egocentric and allocentric boundary coding neurons.

    Args:
        n_hd_neurons (int): Number of head direction (HD) neurons.
        tr_res (float): Resolution of rotated versions of the environment.
        segment_res (float): Resolution for environmental boundary segments.
        r_max (float): Maximum radius for the polar grid.
        polar_dist_res (float): Polar distance resolution.
        n_radial_points (int): Number of radial points in the polar grid.
        polar_ang_res (float): Polar angle resolution.
        sigma_angular (float): Width parameter for angular Gaussian functions.
        amplitude_max (float, optional): Maximum amplitude for the Gaussian functions (default: 1).
        sparseness (int, optional): Sparseness constraint for weight matrices (How many connections to spare) (default: 18,000).
        bvc_tc_clip (float, optional): Clip value for BVC to TC weights (How many percent of weights to clip, should be a value from 0 to 1) (default: 0.01).
        tc_pw_clip (float, optional): Clip value for TC to PW weights (How many percent of weights to clip, should be a value from 0 to 1) (default: 0.01).
        n_steps (int, optional): Number of steps for generating weights (default: 400,000).

    Attributes:
        n_hd_neurons (int): Number of head direction (HD) neurons.
        tr_res (float): Resolution for transformation circuit (TC) representation.
        segment_res (float): Resolution for environmental boundary segments.
        r_max (float): Maximum radius for the polar grid.
        polar_dist_res (float): Polar distance resolution.
        n_radial_points (int): Number of radial points in the polar grid.
        polar_ang_res (float): Polar angle resolution.
        sigma_angular (float): Width parameter for angular Gaussian functions.
        amplitude_max (float): Maximum amplitude for the Gaussian functions.
        sparseness (int): Sparseness constraint for weight matrices (How many connections to spare).
        bvc_tc_clip (float): Clip value for BVC to TC weights (How many percent of weights to clip, should be a value from 0 to 1).
        tc_pw_clip (float): Clip value for TC to PW weights (How many percent of weights to clip, should be a value from 0 to 1).
        n_steps (int): Number of steps for generating weights.
        n_tr (int): Number of layers in transformation circuit based on tr_res.
        tr_angles (np.ndarray): Array of head direction angles.
        n_bvc (int): Total number of BVC neurons.

    Methods:
        get_grid_activity(environment: np.ndarray) -> np.ndarray:
            Calculates grid cell activities in the parietal window (PW) based on environmental boundaries.

        get_hd_activity(bump_locations: np.ndarray) -> np.ndarray:
            Calculates head direction (HD) cell activities based on bump locations.

        initialize_hd2tr_weights() -> np.ndarray:
            Initializes the weights connecting HD neurons to the TC neurons.

        initialize_tr2pw_bvc2tr_weights() -> Tuple[np.ndarray, np.ndarray]:
            Initializes weights connecting TC neurons to PW and BVC neurons to TC neurons.

        invert_weights(*weights: np.ndarray) -> Tuple[np.ndarray, ...]:
            Inverts the specified weight matrices.

        maxnorm(*weights: np.ndarray) -> Tuple[np.ndarray, ...]:
            Normalizes weight matrices using max-norm scaling.

        sumnorm(*weights: np.ndarray) -> Tuple[np.ndarray, ...]:
            Normalizes weight matrices by summing along specific dimensions.

        sparse_weights(tp2pw_weights: np.ndarray, pw2tr_weights: np.ndarray, bvc2tr_weights: np.ndarray, tr2bvc_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Applies sparseness constraints to the weight matrices and clips values accordingly.

        generate() -> TensorGroup:
            Generates and normalizes neural connectivity for the TC in the BVC network model.
    """
    def __init__(
        self,
        n_hd_neurons: int,
        tr_res: float,
        segment_res: float,
        r_max: float,
        polar_dist_res: float,
        n_radial_points: int,
        polar_ang_res: float,
        sigma_angular: float,
        amplitude_max: float = 1,
        sparseness: int = 18_000,
        bvc_tc_clip: float = 0.01,
        tc_pw_clip: float = 0.01,
        n_steps: int = 400_000,

    ):
        """
        Initialize the TCGenerator.

        Args:
            n_hd_neurons (int): Number of head direction neurons.
            tr_res (float): Resolution for transformation circuit (TC) representation.
            segment_res (float): Resolution of segments.
            r_max (float): Maximum radius for polar coordinates.
            polar_dist_res (float): Resolution for polar distance.
            n_radial_points (int): Number of radial points.
            polar_ang_res (float): Resolution for polar angle.
            sigma_angular (float): Standard deviation for angular HD activity.
            amplitude_max (float, optional): Maximum amplitude of activity. Defaults to 1.
            sparseness (int, optional): Sparseness threshold (How many connections to spare). Defaults to 18,000.
            bvc_tc_clip (float, optional): Threshold (Percentage) for clipping BVC to TC activity. Defaults to 0.01.
            tc_pw_clip (float, optional): Threshold (Percentage) for clipping TC to PW activity. Defaults to 0.01.
            n_steps (int, optional): Number of steps. Defaults to 400,000.

        """
        self.n_hd_neurons = n_hd_neurons
        self.tr_res = tr_res
        self.segment_res = segment_res
        self.r_max = r_max
        self.polar_dist_res = polar_dist_res
        self.n_radial_points = n_radial_points
        self.polar_ang_res = polar_ang_res
        self.sigma_angular = sigma_angular
        self.amplitude_max = amplitude_max
        self.n_steps = n_steps
        self.sparseness = sparseness
        self.bvc_tc_clip = bvc_tc_clip
        self.tc_pw_clip = tc_pw_clip
        self.n_tr = (np.floor(2*np.pi/self.tr_res)).astype(int)
        self.tr_angles = np.arange(0, (self.n_tr)*self.tr_res, self.tr_res)
        n_bvc_r = np.rint(self.r_max/self.polar_dist_res).astype(int) # How many BVC neurons? (Will be same as num of PW neurons and each TR sublayer)
        self.n_bvc_theta = (np.floor((2*np.pi - .01)/self.polar_ang_res) + 1).astype(int)
        self.n_bvc = (n_bvc_r * self.n_bvc_theta).astype(int)
        self.polar_distance, self.polar_angle = self.__init_polar_params()

    def __init_polar_params(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate polar parameters for BVC neurons.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing polar distances and polar angles.
        """
        polar_distance = calculate_polar_distance(self.r_max)
        polar_angle = np.arange(0, self.n_bvc_theta * self.polar_ang_res, self.polar_ang_res)
        polar_distance, polar_angle = np.meshgrid(polar_distance, polar_angle)
        return polar_distance, polar_angle

    def get_grid_activity(
        self,
        environment: np.ndarray
    ) -> np.ndarray:
        """
        Calculate grid cell activity based on the environment.

        Args:
            environment (np.ndarray): The environment represented as line segments
            (numpy 2D matrix with columns corresponding to (x_start, y_start, x_end, y_end) start and end coordinates of edge respectively).

        Example:
            >>> tc_geneator.get_grid_activity(np.array([[0, 1, 1, 0], [1, 0, 0, 1]]))

        Returns:
            np.ndarray: Grid cell activity.

        """
        grid_distance = self.polar_distance.flatten()
        grid_angle = self.polar_angle.flatten()
        grid_angle[grid_angle > np.pi] -= 2 * np.pi

        n_points = int(round((2*self.r_max) / self.segment_res))
        x = np.arange(-self.r_max + self.segment_res / 2, -self.r_max + (n_points - 0.5) * self.segment_res + self.segment_res, self.segment_res)
        x, y = np.meshgrid(x, x.copy())

        grid_activation = np.zeros(self.n_bvc)
        for boundary_number in range(environment.shape[0]):
            xi = environment[boundary_number, 0]
            xf = environment[boundary_number, 2]
            yi = environment[boundary_number, 1]
            yf = environment[boundary_number, 3]
            den = np.sqrt((xf - xi) ** 2 + (yf - yi) ** 2)
            len_x = (xf - xi) / den
            len_y = (yf - yi) / den

            grid_perpendicular_displacement_x = -(x - xi) * (1 - len_x ** 2) + (y - yi) * len_y * len_x
            grid_perpendicular_displacement_y = -(y - yi) * (1 - len_y ** 2) + (x - xi) * len_x * len_y

            if xf != xi:
                t = (x + grid_perpendicular_displacement_x - xi) / (xf - xi)
            else:
                t = (y + grid_perpendicular_displacement_y - yi) / (yf - yi)

            # FIXME: Added a constant value 1e-7 for numerical stability
            boundary_points = (t >= 0) & (t <= 1) & \
                (grid_perpendicular_displacement_x >= -self.segment_res / 2) & (grid_perpendicular_displacement_x < self.segment_res / 2) & \
                (grid_perpendicular_displacement_y >= -self.segment_res / 2 + 1e-7) & (grid_perpendicular_displacement_y < self.segment_res / 2 + 1e-7) & \
                (grid_perpendicular_displacement_y**2 + grid_perpendicular_displacement_x**2 <= (self.segment_res/2)**2 + 1e-7)

            boundary_points_x = x[boundary_points]
            boundary_points_y = y[boundary_points]
            boundary_theta, boundary_r = np.arctan2(boundary_points_y, boundary_points_x), np.hypot(boundary_points_x, boundary_points_y)

            for boundary_number_ in range(boundary_theta.shape[0]):
                grid_activation += get_boundary_activations(
                    grid_angle,
                    boundary_theta[boundary_number_],
                    grid_distance,
                    boundary_r[boundary_number_],
                )

        maximum = np.max(grid_activation)

        if maximum > 0.0:
            grid_activation /= maximum  # Normalize

        return grid_activation

    def get_hd_activity(
        self,
        bump_locations: np.ndarray
    ) -> np.ndarray:
        """
        Calculate head direction activity based on bump locations.

        Args:
            bump_locations (np.ndarray): Array of bump locations (1D array of head direction angles).

        Returns:
            np.ndarray: Head direction activity.

        """
        # Ensure bump_locations is in the range [0, 2*pi) and scale it
        bump_locations = self.n_hd_neurons * bump_locations / (2 * np.pi)
        activity = np.zeros(self.n_hd_neurons)
        x = triple_arange(1, self.n_hd_neurons + 1)

        for bump in range(bump_locations.shape[0]):
            x0 = bump_locations[bump]
            g = triple_gaussian(self.amplitude_max, x, x0, self.n_hd_neurons, self.sigma_angular)
            activity += g[self.n_hd_neurons:2 * self.n_hd_neurons]

        return activity

    def initialize_hd2tr_weights(self) -> np.ndarray:
        """
        Initialize weights between head direction neurons and transformation circuit neurons.

        Returns:
            np.ndarray: Initialized weights.

        """
        hd2tr_weights = np.zeros((self.n_bvc, self.n_hd_neurons, self.n_tr))
        for i in range(self.n_tr):
            head_directions = self.tr_angles[i]
            hd_rate = self.get_hd_activity(np.array([head_directions]))
            hd_rate[hd_rate < 0.01] = 0
            hd_rate = csr_matrix(hd_rate)
            hd2tr_weights[:, :, i] = np.outer(np.ones(self.r_max*self.n_radial_points), hd_rate.toarray())

        return hd2tr_weights

    def initialize_tr2pw_bvc2tr_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize weights between transformation circuit and PW/BVC neurons.

        Returns:
            tuple[np.ndarray, np.ndarray]: Initialized weights for TR to PW and BVC to TR.

        """
        tr2pw_weights = np.zeros((self.n_bvc, self.n_bvc, self.n_tr))
        bvc2tr_weights = np.zeros((self.n_bvc, self.n_bvc))

        for count in range(1, self.n_steps + 1):
            # Generate random edge
            theta_i = 2 * np.pi * np.random.rand()
            dist_i = self.r_max * np.random.rand()
            theta_f = 2 * np.pi * np.random.rand()
            # Pick a random HD from TRangles
            tr_layer = np.random.randint(0, self.n_tr)

            xi, yi = dist_i * np.cos(theta_i), dist_i * np.sin(theta_i)
            xf, yf = xi + dist_i * np.cos(theta_f), yi + dist_i * np.sin(theta_f)


            # Generate BVC grid activity for the edge
            bvc_rate = self.get_grid_activity(np.array([[xi, yi, xf, yf],]))
            head_direction = self.tr_angles[tr_layer]

            # Generate the rotated edge
            rxi, ryi = xi * np.cos(head_direction) + yi * np.sin(head_direction), -xi * np.sin(head_direction) + yi * np.cos(head_direction)
            rxf, ryf = xf * np.cos(head_direction) + yf * np.sin(head_direction), -xf * np.sin(head_direction) + yf * np.cos(head_direction)

            # Generate PW layer activity
            pw_rate = self.get_grid_activity(np.array([[rxi, ryi, rxf, ryf],]))

            # Weight Updates
            if count % self.n_tr == 0:
                bvc2tr_weights += np.outer(bvc_rate, bvc_rate)

            tr2pw_weights[:, :, tr_layer] += np.outer(pw_rate, bvc_rate)

        return tr2pw_weights, bvc2tr_weights

    @staticmethod
    def invert_weights(*weights: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Invert weight matrices.

        Args:
            *weights (np.ndarray): Weight matrices to be inverted.

        Returns:
            Tuple[np.ndarray, ...]: Tuple of inverted weight matrices.
        """
        out = list()
        for weight in weights:
            if weight.ndim <= 2:
                out.append(weight.T)
            else:
                out.append(np.transpose(weight, (1, 0, -1)))
        return tuple(out)

    @staticmethod
    def maxnorm(*weights: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Normalize weight matrices using max norm (along first two dimensions).

        Args:
            *weights (np.ndarray): Weight matrices to be normalized.

        Returns:
            Tuple[np.ndarray, ...]: Tuple of normalized weight matrices.

        """
        return (weight/np.max(weight, axis=(0, 1), keepdims=True) for weight in weights)

    @staticmethod
    def sumnorm(*weights: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Normalize weight matrices by sum along 2nd dimension.

        Args:
            *weights (np.ndarray): Weight matrices to be normalized.

        Returns:
            Tuple[np.ndarray, ...]: Tuple of normalized weight matrices.

        """
        out = list()
        for weight in weights:
            scale = np.sum(weight, axis=1, keepdims=True)
            scale[scale == 0] = 1
            out.append(weight/scale)

        return tuple(out)

    def sparse_weights(
        self,
        tr2pw_weights: np.ndarray,
        pw2tr_weights: np.ndarray,
        bvc2tr_weights: np.ndarray,
        tr2bvc_weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply sparseness constraint to weight matrices.

        Args:
            tp2pw_weights (np.ndarray): TR to PW weights.
            pw2tr_weights (np.ndarray): PW to TR weights.
            bvc2tr_weights (np.ndarray): BVC to TR weights.
            tr2bvc_weights (np.ndarray): TR to BVC weights.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of sparse weight matrices.

        """
        tp_clip = self.tc_pw_clip
        bt_clip = self.bvc_tc_clip
        for i in range(self.n_tr):
            tr_sparseness = np.sum(tr2pw_weights[:, :, i] > 0)  # sparseness

            while tr_sparseness > self.sparseness:
                tp_clip += 0.01
                tr2pw_weights[:, :, i] *= (tr2pw_weights[:, :, i] > tp_clip)
                tr_sparseness = np.sum(tr2pw_weights[:, :, i] > 0)

            pw2tr_weights[:, :, i] *= (pw2tr_weights[:, :, i] > tp_clip)

        bt_sparseness = np.sum(bvc2tr_weights > 0)  # sparseness
        while bt_sparseness > self.sparseness:
            bt_clip += 0.01
            bvc2tr_weights *= (bvc2tr_weights > bt_clip)
            bt_sparseness = np.sum(bvc2tr_weights > 0)

        tr2bvc_weights *= (tr2bvc_weights > bt_clip)

        return tr2pw_weights, pw2tr_weights, bvc2tr_weights, tr2bvc_weights

    def generate(self) -> TensorGroup:
        """
        Generate neural model weights.

        Returns:
            TensorGroup: An instance of the TensorGroup class representing the generated weights.

        """
        tr2pw_weights, bvc2tr_weights = self.initialize_tr2pw_bvc2tr_weights()
        hd2tr_weights = self.initialize_hd2tr_weights()
        tr2pw_weights, bvc2tr_weights, hd2tr_weights = self.maxnorm(
            tr2pw_weights, bvc2tr_weights, hd2tr_weights
        )
        pw2tr_weights, tr2bvc_weights = self.invert_weights(tr2pw_weights, bvc2tr_weights)
        tr2pw_weights, pw2tr_weights, bvc2tr_weights, tr2bvc_weights = self.sparse_weights(
            tr2pw_weights, pw2tr_weights, bvc2tr_weights, tr2bvc_weights
        )
        tr2pw_weights, pw2tr_weights, bvc2tr_weights, tr2bvc_weights = self.sumnorm(
            tr2pw_weights, pw2tr_weights, bvc2tr_weights, tr2bvc_weights
        )

        return TensorGroup(
            DirectedTensor(
                from_='tr',
                to='pw',
                weights=tr2pw_weights,
            ),
            DirectedTensor(
                from_='bvc',
                to='tr',
                weights=bvc2tr_weights,
            ),
            DirectedTensor(
                from_='hd',
                to='tr',
                weights=hd2tr_weights,
            ),
            DirectedTensor(
                from_='pw',
                to='tr',
                weights=pw2tr_weights,
            ),
            DirectedTensor(
                from_='tr',
                to='bvc',
                weights=tr2bvc_weights,
            )
        )

