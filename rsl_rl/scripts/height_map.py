import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union


class SceneItems3D:
    """Store and manage 3D geometries (axis-aligned boxes and spheres)."""

    def __init__(self):
        self.boxes: List[List[float]] = []    # [cx, cy, cz, sx, sy, sz]
        self.spheres: List[List[float]] = []  # [cx, cy, cz, r]

    def add_box(self, position: Tuple[float, float, float], size_xyz: Tuple[float, float, float]):
        cx, cy, cz = position
        sx, sy, sz = size_xyz
        self.boxes.append([cx, cy, cz, sx, sy, sz])
        return self

    def add_sphere(self, position: Tuple[float, float, float], radius: float):
        cx, cy, cz = position
        self.spheres.append([cx, cy, cz, radius])
        return self

    def clear(self):
        self.boxes.clear()
        self.spheres.clear()


class HeightMapGenerator:
    """Precompute a height map over a user-specified XY grid and serve fast queries."""

    def __init__(self, scene: SceneItems3D, ground_z: float = 0.0):
        self.scene = scene
        self.ground_z = float(ground_z)

        # Precomputed map storage
        self._H: Optional[np.ndarray] = None     # shape [Ny, Nx]
        self._X: Optional[np.ndarray] = None     # shape [Ny, Nx]
        self._Y: Optional[np.ndarray] = None     # shape [Ny, Nx]
        self._meta: Optional[dict] = None        # grid meta data

    def reset_map(self):
        """Clear any precomputed map."""
        self._H = None
        self._X = None
        self._Y = None
        self._meta = None

    def generate_map(
        self,
        top_left: Tuple[float, float],
        size_xy: Tuple[float, float],
        resolution: Union[Tuple[float, float], float],
    ):
        """Precompute the height map on a regular grid."""
        x_left, y_top = float(top_left[0]), float(top_left[1])
        width_x, height_y = float(size_xy[0]), float(size_xy[1])

        if isinstance(resolution, (tuple, list, np.ndarray)):
            res_x, res_y = float(resolution[0]), float(resolution[1])
        else:
            res_x = res_y = float(resolution)

        if res_x <= 0 or res_y <= 0:
            raise ValueError("Resolution must be positive.")

        Nx = int(np.floor(width_x / res_x)) + 1
        Ny = int(np.floor(height_y / res_y)) + 1
        if Nx <= 1 or Ny <= 1:
            raise ValueError("Grid must have at least 2x2 samples.")

        xs = x_left + np.arange(Nx, dtype=float) * res_x        # [Nx]
        ys = y_top - np.arange(Ny, dtype=float) * res_y         # [Ny], decreasing

        X, Y = np.meshgrid(xs, ys)                              # [Ny, Nx]
        H = np.full((Ny, Nx), self.ground_z, dtype=float)

        # Boxes
        for (cx, cy, cz, sx, sy, sz) in self.scene.boxes:
            inside = (np.abs(X - cx) <= sx * 0.5) & (np.abs(Y - cy) <= sy * 0.5)
            top_z = cz + sz * 0.5
            H = np.where(inside, np.maximum(H, top_z), H)

        # Spheres
        for (cx, cy, cz, r) in self.scene.spheres:
            DX = X - cx
            DY = Y - cy
            d2 = DX * DX + DY * DY
            inside = d2 <= (r * r)
            cap = cz + np.sqrt(np.maximum(r * r - d2, 0.0))
            H = np.where(inside, np.maximum(H, cap), H)

        # Store
        self._H, self._X, self._Y = H, X, Y
        self._meta = dict(
            x_left=x_left, y_top=y_top,
            width_x=width_x, height_y=height_y,
            res_x=res_x, res_y=res_y,
            Nx=Nx, Ny=Ny,
            x_right=xs[-1],
            y_bottom=ys[-1],
            eps=1e-9,
        )

    def get_full_map(self):
        """Return the full precomputed map (H, X, Y, meta)."""
        self._require_map()
        return self._H.copy(), self._X.copy(), self._Y.copy(), dict(self._meta)

    def _require_map(self):
        if self._H is None or self._meta is None:
            raise RuntimeError("Height map not generated. Call generate_map(...) first.")

    def query_point(self, xy: Tuple[float, float], method: str = "nearest") -> float:
        """Query height at a single (x, y) from the precomputed map.

        method: 'nearest' or 'bilinear'
        """
        self._require_map()
        x, y = float(xy[0]), float(xy[1])
        m = self._meta

        # Convert to fractional indices
        fx = (x - m["x_left"]) / m["res_x"]
        fy = (m["y_top"] - y) / m["res_y"]

        if method.lower() == "nearest":
            ix = int(np.round(fx))
            iy = int(np.round(fy))
            if ix < 0 or iy < 0 or ix >= m["Nx"] or iy >= m["Ny"]:
                raise ValueError("Query out of precomputed map bounds.")
            return float(self._H[iy, ix])

        elif method.lower() == "bilinear":
            # Clamp to valid interior range for interpolation
            fx = np.clip(fx, 0.0, m["Nx"] - 1 - m["eps"])
            fy = np.clip(fy, 0.0, m["Ny"] - 1 - m["eps"])
            ix0 = int(np.floor(fx))
            iy0 = int(np.floor(fy))
            ix1 = min(ix0 + 1, m["Nx"] - 1)
            iy1 = min(iy0 + 1, m["Ny"] - 1)
            tx = fx - ix0
            ty = fy - iy0

            h00 = self._H[iy0, ix0]
            h10 = self._H[iy0, ix1]
            h01 = self._H[iy1, ix0]
            h11 = self._H[iy1, ix1]
            return float(
                (1 - tx) * (1 - ty) * h00 +
                tx * (1 - ty) * h10 +
                (1 - tx) * ty * h01 +
                tx * ty * h11
            )
        else:
            raise ValueError("method must be 'nearest' or 'bilinear'.")

    def query_grid(
        self,
        top_left: Tuple[float, float],
        size_xy: Tuple[float, float],
        resolution: Union[Tuple[float, float], float],
        order: str = "x-major",
        return_coords: bool = False,
        rotate_offset: float = 0.0,
    ):
        """Query a grid window from the precomputed map (fast slicing).
        
        Args:
            top_left: Top-left corner of the query window (x, y)
            size_xy: Size of the window (width_x, height_y)
            resolution: Grid resolution (uniform or (res_x, res_y))
            order: Flattening order - 'x-major' or 'y-major'
            return_coords: If True, return coordinates as well
            rotate_offset: Z-axis rotation angle in radians (user's robotics frame).
                          Positive rotation is CCW in user's frame.
                          The sampling grid will be rotated by -rotate_offset in the
                          height map's image frame before sampling.
        """
        self._require_map()
        m = self._meta

        if isinstance(resolution, (tuple, list, np.ndarray)):
            res_x, res_y = float(resolution[0]), float(resolution[1])
        else:
            res_x = res_y = float(resolution)

        x0, y0 = float(top_left[0]), float(top_left[1])
        w, h = float(size_xy[0]), float(size_xy[1])

        Nxw = int(np.floor(w / res_x)) + 1
        Nyw = int(np.floor(h / res_y)) + 1
        if Nxw <= 1 or Nyw <= 1:
            raise ValueError("Requested window must have at least 2x2 samples.")

        # If no rotation, use fast path (existing implementation)
        if abs(rotate_offset) < 1e-9:
            if abs(res_x - m["res_x"]) > m["eps"] or abs(res_y - m["res_y"]) > m["eps"]:
                raise ValueError("Requested resolution must match the precomputed map resolution.")

            # Fractional indices in the precomputed grid
            fx0 = (x0 - m["x_left"]) / m["res_x"]
            fy0 = (m["y_top"] - y0) / m["res_y"]

            # Check alignment to integer indices
            tol = 1e-6
            if abs(fx0 - round(fx0)) > tol or abs(fy0 - round(fy0)) > tol:
                raise ValueError("Requested window top-left is not aligned to the precomputed grid.")

            ix0 = int(round(fx0))
            iy0 = int(round(fy0))

            ix1 = ix0 + Nxw - 1
            iy1 = iy0 + Nyw - 1

            if ix0 < 0 or iy0 < 0 or ix1 >= m["Nx"] or iy1 >= m["Ny"]:
                raise ValueError("Requested window is out of precomputed map bounds.")

            Hw = self._H[iy0:iy1 + 1, ix0:ix1 + 1]
            Xw = self._X[iy0:iy1 + 1, ix0:ix1 + 1] if return_coords else None
            Yw = self._Y[iy0:iy1 + 1, ix0:ix1 + 1] if return_coords else None

            order = order.lower()
            if order in ("x-major", "xy", "c"):
                flat = Hw.ravel(order="C")
            elif order in ("y-major", "yx", "f"):
                flat = Hw.ravel(order="F")
            else:
                raise ValueError("order must be 'x-major' or 'y-major'.")

            if return_coords:
                return flat, Xw, Yw, Hw
            return flat

        # Rotated path: generate grid, rotate, sample with interpolation
        # Generate unrotated grid points
        xs = x0 + np.arange(Nxw, dtype=float) * res_x
        ys = y0 - np.arange(Nyw, dtype=float) * res_y  # decreasing (image frame)
        X_grid, Y_grid = np.meshgrid(xs, ys)  # [Nyw, Nxw]
        
        # Find center of the grid
        center_x = x0 + w * 0.5
        center_y = y0 - h * 0.5
        
        # Rotate points around center by rotate_offset
        # Since we're rotating the sampling grid (not the map), and both frames
        # share the x-axis, we apply the rotation with the same sign
        theta = rotate_offset
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Translate to origin, rotate, translate back
        dx = X_grid - center_x
        dy = Y_grid - center_y
        X_rotated = center_x + cos_theta * dx - sin_theta * dy
        Y_rotated = center_y + sin_theta * dx + cos_theta * dy
        
        # Sample heights at rotated positions using vectorized bilinear interpolation
        # Convert rotated positions to fractional indices
        fx = (X_rotated - m["x_left"]) / m["res_x"]
        fy = (m["y_top"] - Y_rotated) / m["res_y"]
        
        # Clamp to valid interior range for interpolation
        fx = np.clip(fx, 0.0, m["Nx"] - 1 - m["eps"])
        fy = np.clip(fy, 0.0, m["Ny"] - 1 - m["eps"])
        
        # Get integer indices for the four surrounding points
        ix0 = np.floor(fx).astype(int)
        iy0 = np.floor(fy).astype(int)
        ix1 = np.minimum(ix0 + 1, m["Nx"] - 1)
        iy1 = np.minimum(iy0 + 1, m["Ny"] - 1)
        
        # Compute interpolation weights
        tx = fx - ix0
        ty = fy - iy0
        
        # Gather heights at the four corners
        h00 = self._H[iy0, ix0]
        h10 = self._H[iy0, ix1]
        h01 = self._H[iy1, ix0]
        h11 = self._H[iy1, ix1]
        
        # Bilinear interpolation
        Hw = ((1 - tx) * (1 - ty) * h00 +
              tx * (1 - ty) * h10 +
              (1 - tx) * ty * h01 +
              tx * ty * h11)
        
        # Prepare coordinate grids if needed
        Xw = X_rotated if return_coords else None
        Yw = Y_rotated if return_coords else None
        
        # Flatten according to order
        order = order.lower()
        if order in ("x-major", "xy", "c"):
            flat = Hw.ravel(order="C")
        elif order in ("y-major", "yx", "f"):
            flat = Hw.ravel(order="F")
        else:
            raise ValueError("order must be 'x-major' or 'y-major'.")
        
        if return_coords:
            return flat, Xw, Yw, Hw
        return flat


def _draw_full_map(H: np.ndarray, X: np.ndarray, Y: np.ndarray, title: str = "Full height map"):
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, H, shading="auto", cmap="viridis")
    plt.colorbar(label="Height (m)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.2)


def _draw_window(Hw: np.ndarray, Xw: np.ndarray, Yw: np.ndarray, title: str = "Window height map"):
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(Xw, Yw, Hw, shading="auto", cmap="magma")
    plt.colorbar(label="Height (m)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.2)


if __name__ == "__main__":
    # Example usage
    scene = SceneItems3D()
    scene.add_box((0.5, 0.0, 0.5), (1.5, 1.0, 1.0))
    scene.add_sphere((2.0, 1.0, 0.6), 0.75)

    gen = HeightMapGenerator(scene, ground_z=0.0)
    gen.generate_map(top_left=(-2.0, 2.0), size_xy=(6.0, 4.0), resolution=0.02)

    H, X, Y, meta = gen.get_full_map()
    _draw_full_map(H, X, Y, title="Full precomputed height map")

    window_top_left = (-1.0, 1.5)
    window_size = (2.0, 1.5)
    flat, Xw, Yw, Hw = gen.query_grid(
        top_left=window_top_left,
        size_xy=window_size,
        resolution=meta["res_x"],
        order="x-major",
        return_coords=True,
    )
    _draw_window(Hw, Xw, Yw, title="Queried window height map")

    print("Nearest height at (0.0, 0.0):", gen.query_point((0.0, 0.0), method="nearest"))
    print("Bilinear height at (0.1, 0.1):", gen.query_point((0.1, 0.1), method="bilinear"))

    plt.tight_layout()
    plt.show()