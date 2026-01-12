from height_map import SceneItems3D

def build_linear_stairs_scene3d(
    center: tuple[float, float] = (8.0, 0.0),
    num_steps: int = 10,
    step_height: float = 0.1,
    step_width: float = 0.3,
    stairs_width: float = 1.0,
    direction: str = "y+",
    base_z: float = 0.0,
) -> SceneItems3D:
    """
    Create a SceneItems3D with a simple linear staircase made of boxes.
    Steps grow from ground (base_z) up to (i+1)*step_height. Each step is a rectangular box.

    direction: 'y+' | 'y-' | 'x+' | 'x-'
    """
    scene3d = SceneItems3D()
    cx0, cy0 = float(center[0]), float(center[1])

    if direction == "y+":
        ux, uy, along_x = 0.0, 1.0, False
    elif direction == "y-":
        ux, uy, along_x = 0.0, -1.0, False
    elif direction == "x+":
        ux, uy, along_x = 1.0, 0.0, True
    elif direction == "x-":
        ux, uy, along_x = -1.0, 0.0, True
    else:
        raise ValueError(f"Unknown direction: {direction}")

    for i in range(int(num_steps)):
        # center of step i is at (i + 0.5) * step_width along direction
        cx = cx0 + (i + 0.5) * step_width * ux
        cy = cy0 + (i + 0.5) * step_width * uy
        # full height from ground to top of the step
        full_h = base_z + (i + 1) * step_height
        cz = full_h * 0.5  # center at half height so bottom rests on ground (z=0)

        if along_x:
            size = (step_width, stairs_width, full_h)
        else:
            size = (stairs_width, step_width, full_h)
        scene3d.add_box((cx, cy, cz), size)

    return scene3d

def _build_run_steps_3d(
    scene3d: SceneItems3D,
    start_xy: tuple[float, float],
    direction: str,          # "y+" | "y-" | "x+" | "x-"
    base_z: float,
    stairs_width: float,
    num_steps: int,
    step_h: float,
    tread: float,
) -> tuple[float, tuple[float, float], float]:
    """
    Place num_steps identical treads of length=tread advancing along 'direction'.
    Step i (0-based) sits at horizontal offset i*tread from start and rises by i*step_h.
    Returns:
      z_gain: num_steps*step_h
      far_edge_xy: far edge after last tread
      run_length: num_steps*tread
    """
    x0, y0 = float(start_xy[0]), float(start_xy[1])

    if direction == "y+":
        ux, uy, along_x = 0.0, 1.0, False
    elif direction == "y-":
        ux, uy, along_x = 0.0, -1.0, False
    elif direction == "x+":
        ux, uy, along_x = 1.0, 0.0, True
    elif direction == "x-":
        ux, uy, along_x = -1.0, 0.0, True
    else:
        raise ValueError(f"Unknown direction: {direction}")

    for i in range(int(num_steps)):
        cx = x0 + (i + 0.5) * tread * ux
        cy = y0 + (i + 0.5) * tread * uy
        full_h = base_z + (i + 1) * step_h
        cz = full_h * 0.5
        size = (tread, stairs_width, full_h) if along_x else (stairs_width, tread, full_h)
        scene3d.add_box((cx, cy, cz), size)

    run_length = num_steps * tread
    far_x = x0 + run_length * ux
    far_y = y0 + run_length * uy
    return num_steps * step_h, (far_x, far_y), run_length


def build_turning_stairs_90_scene3d(
    center: tuple[float, float] = (8.0, 0.0),
    num_steps_run1: int = 8,
    num_steps_run2: int = 6,
    step_height: float = 0.08,
    step_width: float = 0.30,        # tread depth
    stairs_width: float = 1.2,
    landing_length: float = 1.2,
    landing_width: float | None = None,
    turn_right: bool = True,
    base_z: float = 0.0,
) -> SceneItems3D:
    """
    Create a SceneItems3D with a 90-degree turning staircase:
    - Run 1 along +y from the entry edge
    - Landing
    - Run 2 along ±x from the landing
    - Second landing (exit) oriented along x
    All steps and landings are axis-aligned boxes rising from ground_z=base_z.
    """
    scene3d = SceneItems3D()
    cx0, cy0 = float(center[0]), float(center[1])

    # Run 1 (+y)
    run1_start = (cx0, cy0)
    z1, run1_far, run1_len = _build_run_steps_3d(
        scene3d,
        start_xy=run1_start,
        direction="y+",
        base_z=base_z,
        stairs_width=stairs_width,
        num_steps=num_steps_run1,
        step_h=step_height,
        tread=step_width,
    )

    # Landing after run 1
    landing_th = max(step_height * 0.5, 0.02)
    land_w = float(stairs_width if landing_width is None else landing_width)
    landing_center = (run1_far[0], run1_far[1] + 0.5 * landing_length)
    landing_full_h = z1 + landing_th
    scene3d.add_box((landing_center[0], landing_center[1], landing_full_h * 0.5), (land_w, landing_length, landing_full_h))

    # Run 2 along ±x on top of landing
    if turn_right:
        run2_start = (landing_center[0] + 0.5 * stairs_width, landing_center[1])
        dir2 = "x+"
    else:
        run2_start = (landing_center[0] - 0.5 * stairs_width, landing_center[1])
        dir2 = "x-"
    z2, run2_far, run2_len = _build_run_steps_3d(
        scene3d,
        start_xy=run2_start,
        direction=dir2,
        base_z=landing_full_h,
        stairs_width=stairs_width,
        num_steps=num_steps_run2,
        step_h=step_height,
        tread=step_width,
    )

    # Second landing (exit) after run 2
    exit_len = landing_length
    exit_w = land_w
    z_top2 = landing_full_h + z2
    if dir2 == "x+":
        landing2_center = (run2_far[0] + 0.5 * exit_len, run2_start[1])
    else:
        landing2_center = (run2_far[0] - 0.5 * exit_len, run2_start[1])
    landing2_full_h = z_top2 + landing_th
    scene3d.add_box((landing2_center[0], landing2_center[1], landing2_full_h * 0.5), (exit_len, exit_w, landing2_full_h))

    return scene3d