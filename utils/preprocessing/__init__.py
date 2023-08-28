import numpy as np
import os
import configparser

class GridCellFRGenerator:
    """
    A class for generating grid cell firing rate maps based on specified parameters.
    """
    def __init__(self, ini_path: str, save_path: str):
        """
        Initialize the GridCellFRGenerator.

        Args:
            ini_path (str): Path to the configuration INI file.
            save_path (str): Path where generated data will be saved.
        """
        self.config = configparser.ConfigParser()
        self.config.read(ini_path)

        if not os.path.exists(ini_path):
            raise FileNotFoundError('The ini file does not exist: {}'.format(ini_path))
        if not os.path.exists(save_path):
            raise OSError('The save path does not exist: {}'.format(save_path))

        self.n_mod = self.config.getint('Parameters', 'Nmod')
        self.n_per_mod = self.config.getint('Parameters', 'NperMod')
        self.res = self.config.getint('Parameters', 'res')
        self.x_max = self.config.getint('Parameters', 'Xmax')
        self.y_max = self.config.getint('Parameters', 'Ymax')

        self.f_mods = np.array(self.config.get('Frequencies', 'Fmods').split(','), dtype=float)
        self.FAC = np.array(self.config.get('Offsets', 'FAC').split(','), dtype=float)
        self.r_size = np.array(self.config.get('Template', 'Rsize').split(','), dtype=int)
        self.orientations = np.array(self.config.get('Orientations', 'ORIs').split(','), dtype=float)
        self.save_path = save_path

    def generate_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate coordinates and mesh grids.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: X coordinates, Y coordinates, and their mesh grid.
        """
        X = np.arange(0, self.x_max, self.res)
        Y = np.arange(0, self.y_max, self.res)
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


if __name__ == '__main__':
    ini_path = os.path.join('.', 'cfg', 'grid_cells.ini')
    save_path = os.path.join('.', 'data', 'grid_cells')
    generator = GridCellFRGenerator(ini_path, save_path)
    generator.generate(save=True)
