#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, subprocess
from typing import List, Tuple
import json

import pybullet as p
import pybullet_data
import numpy as np

from src.se3_lpvds.src.se3_class import se3_class
from src.se3_lpvds.src.util import load_tools, process_tools, plot_tools
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from src.se3_lpvds.src.se3_class import se3_class
from mpl_toolkits.mplot3d import Axes3D



def setup_bullet(use_gui: bool, timestep: float = 1.0 / 240.0):

    cid = p.connect(p.GUI if use_gui else p.DIRECT)
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(timestep)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf")
    # Enable real-time simulation and physics
    p.setRealTimeSimulation(1)
    
    # Enable default constraint solver
    p.setPhysicsEngineParameter(numSolverIterations=50)
    
    # Enable collision detection
    p.setPhysicsEngineParameter(enableFileCaching=0,
                               contactERP=0.9,
                               contactBreakingThreshold=0.01)
    return cid, plane


def load_panda(base_pos=(0, 0, 0), base_orn=(0, 0, 0, 1)) -> int:

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
    robot = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=base_pos,
        baseOrientation=base_orn,
        useFixedBase=True,
        flags=flags,
    )
    
    # Enable collision detection between all links
    for i in range(p.getNumJoints(robot)):
        p.setCollisionFilterPair(robot, robot, i, i, 1)
    
    return robot


def get_joint_info(robot: int) -> Tuple[List[int], List[float], List[float]]:

    arm_joint_indices = []
    q_min = []
    q_max = []
    n = p.getNumJoints(robot)
    for j in range(n):
        ji = p.getJointInfo(robot, j)
        jtype = ji[2]
        jname = ji[1].decode()
        if jtype == p.JOINT_REVOLUTE and jname.startswith("panda_joint") and len(arm_joint_indices) < 7:
            arm_joint_indices.append(j)
            q_min.append(ji[8])
            q_max.append(ji[9])
    if len(arm_joint_indices) != 7:
        raise RuntimeError("Failed to find 7 Panda arm joints.")
    return arm_joint_indices, q_min, q_max


def find_ee_link(robot: int) -> int:

    ee = None
    n = p.getNumJoints(robot)
    for j in range(n):
        ji = p.getJointInfo(robot, j)
        lname = ji[12].decode()
        if lname == "panda_hand":
            ee = j
            break
    if ee is None:
        ee = n - 1  # fallback to last link
    return ee


def get_dof_order(robot: int) -> Tuple[List[int], dict]:

    dof_joint_indices = []
    n = p.getNumJoints(robot)
    for j in range(n):
        jtype = p.getJointInfo(robot, j)[2]
        if jtype != p.JOINT_FIXED:
            dof_joint_indices.append(j)
    # Map from joint index -> position in DoF vector
    dof_index_of_joint = {j: i for i, j in enumerate(dof_joint_indices)}
    return dof_joint_indices, dof_index_of_joint


def calculate_jacobian(robot: int, ee_link: int, q_dof: List[float]) -> np.ndarray:

    # PyBullet expects vectors of size numDoF (non-fixed joints)
    zero = [0.0] * len(q_dof)
    jac_t, jac_r = p.calculateJacobian(robot, ee_link, [0, 0, 0], q_dof, zero, zero)
    Jv = np.array(jac_t)
    Jw = np.array(jac_r)
    J = np.vstack([Jv, Jw])  # 6 x nJoints
    return J


def manipulability_yoshikawa(J: np.ndarray, joint_cols: List[int]) -> float:
    # Keep only the columns for the 7 arm joints (use full 6x7 Jacobian)
    J7 = J[:, joint_cols]
    s = np.linalg.svd(J7, compute_uv=False)
    # Yoshikawa measure: product of singular values (equivalent to sqrt(det(JJ^T)))
    return float(np.prod(s))


def manipulability_with_sigma_min(J: np.ndarray, joint_cols: List[int]) -> Tuple[float, float]:
    """Return (Yoshikawa manipulability, smallest singular value)."""
    J7 = J[:, joint_cols]
    s = np.linalg.svd(J7, compute_uv=False)
    m = float(np.prod(s))
    sigma_min = float(np.min(s)) if s.size > 0 else float("nan")
    return m, sigma_min


def angle_to_weak_direction(
    J: np.ndarray,
    joint_cols: List[int],
    current_dir: np.ndarray,
) -> Tuple[float, float]:
    """Compute angle (deg) between current EE direction and weakest task-space direction.

    Uses the left singular vector of Jv (linear velocity Jacobian) corresponding to
    the smallest singular value. Also returns the smallest eigenvalue of Jv Jv^T
    (which equals sigma_min^2).
    """
    Jv = J[:3, joint_cols]
    try:
        U, S, _ = np.linalg.svd(Jv, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    if S.size == 0:
        return float("nan"), float("nan")
    min_idx = int(np.argmin(S))
    weak_vec = U[:, min_idx]
    # Normalize vectors
    n_weak = float(np.linalg.norm(weak_vec))
    if n_weak > 0:
        weak_vec = weak_vec / n_weak
    d = np.asarray(current_dir, dtype=float)
    n_d = float(np.linalg.norm(d))
    if n_d == 0:
        return float("nan"), float(S[min_idx] ** 2)
    d = d / n_d
    dotp = float(np.dot(weak_vec, d))
    dotp = max(-1.0, min(1.0, abs(dotp)))
    angle_deg = float(np.degrees(np.arccos(dotp)))
    eig_min = float(S[min_idx] ** 2)
    return angle_deg, eig_min


def smallest_singular_vector_linear(J: np.ndarray, joint_cols: List[int]) -> Tuple[np.ndarray, float]:
    """Return unit left singular vector of Jv with smallest singular value and sigma_min.

    Jv is the 3xN linear part of the Jacobian. Returns (u_min, sigma_min).
    """
    Jv = J[:3, joint_cols]
    try:
        U, S, _ = np.linalg.svd(Jv, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=float), float("nan")
    if S.size == 0:
        return np.zeros(3, dtype=float), float("nan")
    min_idx = int(np.argmin(S))
    u = U[:, min_idx]
    nrm = float(np.linalg.norm(u))
    if nrm > 0:
        u = u / nrm
    return u.astype(float), float(S[min_idx])


def get_full_joint_vector(robot: int, dof_joint_indices: List[int]) -> List[float]:
    import pybullet as p

    q = []
    for j in dof_joint_indices:
        qj = p.getJointState(robot, j)[0]
        q.append(qj)
    return q


def set_arm_positions(robot: int, arm_joint_indices: List[int], q_arm: np.ndarray, kp=1.0, kd=0.1):
    import pybullet as p

    # Reduced gains for smoother motion
    p.setJointMotorControlArray(
        robot,
        arm_joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=q_arm,
        positionGains=[kp] * len(arm_joint_indices),
        velocityGains=[kd] * len(arm_joint_indices),
        forces=[50.0] * len(arm_joint_indices),  # Reduced forces to allow natural dynamics
    )


def reset_arm(robot: int, arm_joint_indices: List[int], q_arm: np.ndarray):
    import pybullet as p

    for idx, q in zip(arm_joint_indices, q_arm):
        p.resetJointState(robot, idx, float(q))


def angular_sigma_min(J: np.ndarray, joint_cols: List[int]) -> float:
    """Smallest singular value of the rotational (angular) Jacobian block Jw."""
    # Rotational block is the bottom 3 rows
    Jw = J[3:6, joint_cols]
    s = np.linalg.svd(Jw, compute_uv=False)
    return float(np.min(s)) if s.size > 0 else float("nan")


def compute_ik(robot, ee_link, arm_joint_indices, dof_joint_indices, p_target, q_target, current_q=None):
    # Set IK parameters
    max_iters = 100
    residual_threshold = 1e-4
    
    # Use current joint angles as initial guess if provided
    if current_q is None:
        current_q = [0.0] * len(dof_joint_indices)
    else:
        current_q.extend([0.0, 0.0])

    ik_sol = p.calculateInverseKinematics(
        bodyUniqueId=robot,
        endEffectorLinkIndex=ee_link,
        targetPosition=p_target.tolist(),
        targetOrientation=tuple(q_target.as_quat().tolist()),
        currentPositions=current_q,
        maxNumIterations=max_iters,
        residualThreshold=residual_threshold
    )

    q = [ik_sol[d] for d in [dof_joint_indices.index(j) for j in arm_joint_indices]]
    
    # Validate IK solution
    reset_arm(robot, arm_joint_indices, q)
    ee_state = p.getLinkState(robot, ee_link, computeForwardKinematics=True)
    actual_pos = np.array(ee_state[4])
    actual_orn = np.array(ee_state[5])
    
    pos_error = np.linalg.norm(actual_pos - p_target)
    orn_error = np.linalg.norm(actual_orn - q_target.as_quat())
    
    # if pos_error > 0.01 or orn_error > 0.1:
    #     print(f"Warning: Large IK error - pos: {pos_error:.4f}, orn: {orn_error:.4f}")
        
    return q


def draw_waypoints(points: np.ndarray, point_color=(0.9, 0.3, 0.1), line_color=(1, 0, 0), point_size: float = 5.0, line_width: float = 2.0, lifeTime: float = 0.0):
    """Draw waypoints as debug points plus connecting lines in PyBullet.
    - points: (N, 3) array-like in world coordinates
    - point_color/line_color: RGB in [0,1]
    """
    import pybullet as p
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    p.addUserDebugPoints(pointPositions=pts.tolist(), pointColorsRGB=[point_color] * len(pts), pointSize=point_size, lifeTime=lifeTime)



def learn_ds():
    '''Load data'''
    folder_path = '/Users/macpro/Desktop/Project/infeasible_task2_0214/all.mat'
    p_raw, q_raw, t_raw, dt = load_tools._process_bag(folder_path, if_flip=True)

    # p_raw, q_raw, t_raw, dt= load_tools.load_demo_dataset()

    '''Process data'''
    p_in, q_in, t_in             = process_tools.pre_process(p_raw, q_raw, t_raw,  shift=True, opt= "savgol")
    p_out, q_out                 = process_tools.compute_output(p_in, q_in, t_in)
    p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)
    p_in, q_in, p_out, q_out     = process_tools.rollout_list(p_in, q_in, p_out, q_out)

    '''Run lpvds'''
    se3_obj = se3_class(p_in, q_in, p_out, q_out, p_att, q_att, dt, K_init=4)
    se3_obj.begin()

    
    '''Evaluate results'''
    p_init = p_init[0]
    q_init = R.from_quat(-q_init[0].as_quat())
    p_test, q_test, gamma_pos, gamma_ori, v_test, w_test = se3_obj.sim(p_init, q_init, step_size=0.05)

    '''Plot results'''
    plot_tools.plot_result(p_in, p_test, q_test)

    plot_tools.plot_gamma(gamma_pos.T, title="pos")
    plot_tools.plot_gamma(gamma_ori.T, title="ori")

    plt.show()

    return se3_obj, p_init, q_init, p_att, q_att, p_test, q_test




def calculate_angle_to_svec(v: np.ndarray, svec_min: np.ndarray) -> float:
    """Calculate angle between velocity vector and smallest singular vector.
    
    Args:
        v: Velocity vector (3,)
        svec_min: Smallest singular vector (3,)
    
    Returns:
        float: Angle in degrees, or nan if velocity is zero
    """
    v = v[:,0] if v.ndim > 1 else v
    v_norm = np.linalg.norm(v)
    
    if v_norm > 0:
        v_normalized = v / v_norm
        dot_product = np.dot(v_normalized, svec_min)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure valid arccos input
        return np.degrees(np.arccos(dot_product))
    else:
        return float('nan')
    

def integrate(x, xdot, dt=0.01):
        return x + xdot * dt
    


def detect_singularity(robot, ee_link, dof_joint_indices, arm_dof_cols, curr_velo, i):
    q_dof = get_full_joint_vector(robot, dof_joint_indices)
    J = calculate_jacobian(robot, ee_link, q_dof)
    m, sigma_min = manipulability_with_sigma_min(J, arm_dof_cols)
    svec_min, _sv = smallest_singular_vector_linear(J, arm_dof_cols)
    angle_deg = calculate_angle_to_svec(curr_velo, svec_min)
    modulation_flag = True if sigma_min < 0.05 and angle_deg < 30.0 else False

    print(
        f"step {i}: sigma_min {sigma_min:.6e}, "
        f"svec_min [{svec_min[0]: .3f}, {svec_min[1]: .3f}, {svec_min[2]: .3f}], "
        f"angle: {angle_deg:.2f}°"
        f" -> modulation: {modulation_flag}"
    )

    return modulation_flag, J


def generate_modulated_samples(x: np.ndarray, x_dot: np.ndarray, J: np.ndarray, N: int = 30) -> np.ndarray:
    """Generate modulated samples around a point using given velocity and Jacobian."""
    # Step 1: Generate Gaussian samples around current position
    sigma = 0.05
    Sigma = sigma**2 * np.eye(3)
    X_m = np.random.multivariate_normal(x.flatten(), Sigma, N).T  # (3, N)
    
    # Step 2: Get velocity magnitude
    x_dot_norm = np.linalg.norm(x_dot)
    
    # Step 3: Get principal direction and ensure consistent orientation
    Jv = J[:3, :]  # Linear velocity part of Jacobian
    U, S, _ = np.linalg.svd(Jv, full_matrices=False)
    idx = np.argmax(S)  # Index of largest singular value
    principal_dir = U[:, idx]  # Corresponding left singular vector
    
    # Ensure consistent direction by aligning with current velocity
    if x_dot_norm > 0:
        x_dot_normalized = x_dot.flatten() / x_dot_norm
        # If principal direction points opposite to velocity, flip it
        if np.dot(principal_dir, x_dot_normalized) < 0:
            principal_dir = -principal_dir
    
    # print(f"Principal direction: {principal_dir}, Singular value: {S[idx]}")
    
    # Step 4: Create matrix of principal directions
    principalDirs = np.tile(principal_dir.reshape(3, 1), (1, N))
    
    # Step 5: Compute modulated velocities
    Xdot_m = np.real(principalDirs) * x_dot_norm *3
    
    return X_m, Xdot_m





def run_lmds_demo(x_in: np.ndarray, X_m: np.ndarray, Xdot_m: np.ndarray, modulation_flag=1):
    """Run the lmds_demo C++ executable with modulated samples from JSON."""
    if modulation_flag:    
        modulation_flag = 1
        
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Prepare data as dictionary
    data = {
        'X_m': X_m.T.tolist(),    # Convert to (N,3) format for JSON
        'Xdot_m': Xdot_m.T.tolist()  # Convert to (N,3) format for JSON
    }
    
    # Save as JSON
    data_file = os.path.join(data_dir, 'modulation_data.json')
    with open(data_file, 'w') as f:
        json.dump(data, f)
    
    executable_path = os.path.join(os.getcwd(), 'gp_modulate_cpp', 'bin', 'lmds_demo')
    
    try:
        cmd = [
            executable_path,
            str(x_in[0]),  # x coordinate
            str(x_in[1]),  # y coordinate
            str(x_in[2]),  # z coordinate
            str(modulation_flag)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout:
            # Split output lines and get the last non-empty line
            lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            if not lines:
                print("No output received from lmds_demo")
                return None
                
            # Get last line which should contain the velocity values
            last_line = lines[-1]
            try:
                velocity = np.array([float(v) for v in last_line.split()])
                return velocity
            except ValueError as e:
                print(f"Error parsing velocity values: {e}")
                print(f"Output was: {result.stdout}")
                return None
            
    except Exception as e:
        print(f"Error running lmds_demo: {e}")
        return None


def plot_manipulability_over_time(M_list, time_vector=None, title="Manipulability Analysis Over Time", figsize=(12, 8)):
    """
    Plot manipulability index and smallest eigenvalue over simulation time.
    
    Args:
        M_list: Array of manipulability matrices, shape (N, 3, 3)
        time_vector: Time vector, shape (N,). If None, uses indices
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        fig, (ax1, ax2): Figure and axes objects
    """
    M_list = np.asarray(M_list)
    N = M_list.shape[0]
    
    # Create time vector if not provided
    if time_vector is None:
        time_vector = np.arange(N)
    
    # Compute manipulability indices and smallest eigenvalues
    manipulability_indices = np.zeros(N)
    smallest_eigenvalues = np.zeros(N)
    
    for i in range(N):
        M = M_list[i]  # Shape: (3, 3)
        
        # Compute manipulability index (determinant of M)
        manipulability_indices[i] = np.linalg.det(M)
        
        # Compute smallest eigenvalue
        eigenvals = np.linalg.eigvals(M)
        smallest_eigenvalues[i] = np.min(eigenvals)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot manipulability index
    ax1.plot(time_vector, manipulability_indices, 'b-', linewidth=2, label='Manipulability Index')
    ax1.set_ylabel('Manipulability Index', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title(title, fontsize=14)
    
    # Add horizontal line for low manipulability threshold
    mean_manip = np.mean(manipulability_indices)
    ax1.axhline(y=mean_manip * 0.1, color='r', linestyle='--', alpha=0.7, 
                label=f'Low Threshold ({mean_manip * 0.1:.3f})')
    ax1.legend()
    
    # Plot smallest eigenvalue
    ax2.plot(time_vector, smallest_eigenvalues, 'r-', linewidth=2, label='Smallest Eigenvalue')
    ax2.set_xlabel('Time / Step', fontsize=12)
    ax2.set_ylabel('Smallest Eigenvalue', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add horizontal line for singularity threshold
    ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, 
                label='Singularity Threshold (0.05)')
    ax2.legend()
    
    # Highlight singularity regions
    singularity_mask = smallest_eigenvalues < 0.05
    if np.any(singularity_mask):
        ax2.fill_between(time_vector, 0, np.max(smallest_eigenvalues), 
                        where=singularity_mask, alpha=0.2, color='red',
                        label='Singularity Regions')
        ax2.legend()
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def detect_singularity_batch(M_list, velocities, sigma_threshold=0.05, angle_threshold=30.0):
    """
    Detect singularities from manipulability matrices and velocities.
    
    Args:
        M_list: List or array of manipulability matrices, shape (N, 3, 3)
        velocities: Corresponding velocities, shape (N, 3) or (3, N)
        sigma_threshold: Threshold for smallest eigenvalue to detect singularity
        angle_threshold: Angle threshold in degrees between velocity and smallest eigenvector
        
    Returns:
        modulation_flags: Boolean array indicating which points need modulation, shape (N,)
        sigma_mins: Smallest eigenvalues for each matrix, shape (N,)
        angles_deg: Angles between velocities and smallest eigenvectors in degrees, shape (N,)
    """
    # Ensure consistent shapes
    M_list = np.asarray(M_list)  # Shape: (N, 3, 3)
    velocities = np.asarray(velocities)
    
    # Handle velocity shape
    if velocities.shape[0] == 3 and velocities.shape[1] != 3:
        velocities = velocities.T  # Convert from (3, N) to (N, 3)
    
    N = M_list.shape[0]
    sigma_mins = np.zeros(N)
    angles_deg = np.zeros(N)
    modulation_flags = np.zeros(N, dtype=bool)
    
    for i in range(N):
        M = M_list[i]  # Shape: (3, 3)
        velocity = velocities[i]  # Shape: (3,)
        
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(M)
        
        # Find smallest eigenvalue and corresponding eigenvector
        min_idx = np.argmin(eigenvals)
        sigma_min = eigenvals[min_idx]
        eigenvec_min = eigenvecs[:, min_idx]
        
        # Compute angle between velocity and smallest eigenvector
        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm > 1e-8:  # Avoid division by zero
            cos_angle = np.dot(velocity, eigenvec_min) / velocity_norm
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
            angle_rad = np.arccos(np.abs(cos_angle))  # Use absolute value for acute angle
            angle_deg = np.degrees(angle_rad)
        else:
            angle_deg = 0.0
        
        # Store results
        sigma_mins[i] = sigma_min
        angles_deg[i] = angle_deg
        
        # Determine if modulation is needed
        modulation_flags[i] = (sigma_min < sigma_threshold) and (angle_deg < angle_threshold)
    
    return modulation_flags, sigma_mins, angles_deg


def plot_manipulability_statistics(M_list, sigma_mins, angles_deg, modulation_flags, 
                                  title="Manipulability Statistics", figsize=(15, 10)):
    """
    Plot comprehensive manipulability statistics including histograms and scatter plots.
    
    Args:
        M_list: Array of manipulability matrices, shape (N, 3, 3)
        sigma_mins: Smallest eigenvalues, shape (N,)
        angles_deg: Angles between velocities and smallest eigenvectors, shape (N,)
        modulation_flags: Boolean array indicating modulation points, shape (N,)
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        fig, axes: Figure and axes objects
    """
    M_list = np.asarray(M_list)
    N = M_list.shape[0]
    
    # Compute manipulability indices
    manipulability_indices = np.array([np.linalg.det(M) for M in M_list])
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 1. Manipulability index histogram
    axes[0, 0].hist(manipulability_indices, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(np.mean(manipulability_indices), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(manipulability_indices):.4f}')
    axes[0, 0].set_xlabel('Manipulability Index')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Manipulability Index Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Smallest eigenvalue histogram
    axes[0, 1].hist(sigma_mins, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(0.05, color='red', linestyle='--', label='Threshold: 0.05')
    axes[0, 1].axvline(np.mean(sigma_mins), color='orange', linestyle='--', 
                      label=f'Mean: {np.mean(sigma_mins):.4f}')
    axes[0, 1].set_xlabel('Smallest Eigenvalue')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Smallest Eigenvalue Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Angle histogram
    axes[0, 2].hist(angles_deg, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 2].axvline(30.0, color='red', linestyle='--', label='Threshold: 30°')
    axes[0, 2].axvline(np.mean(angles_deg), color='orange', linestyle='--', 
                      label=f'Mean: {np.mean(angles_deg):.1f}°')
    axes[0, 2].set_xlabel('Angle (degrees)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Velocity-Eigenvector Angle Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Scatter: Manipulability vs Smallest Eigenvalue
    scatter1 = axes[1, 0].scatter(manipulability_indices[~modulation_flags], 
                                 sigma_mins[~modulation_flags], 
                                 c='blue', alpha=0.6, label='No Modulation')
    if np.any(modulation_flags):
        scatter2 = axes[1, 0].scatter(manipulability_indices[modulation_flags], 
                                     sigma_mins[modulation_flags], 
                                     c='red', alpha=0.8, label='Modulation Needed')
    axes[1, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Manipulability Index')
    axes[1, 0].set_ylabel('Smallest Eigenvalue')
    axes[1, 0].set_title('Manipulability vs Smallest Eigenvalue')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Scatter: Smallest Eigenvalue vs Angle
    axes[1, 1].scatter(sigma_mins[~modulation_flags], angles_deg[~modulation_flags], 
                      c='blue', alpha=0.6, label='No Modulation')
    if np.any(modulation_flags):
        axes[1, 1].scatter(sigma_mins[modulation_flags], angles_deg[modulation_flags], 
                          c='red', alpha=0.8, label='Modulation Needed')
    axes[1, 1].axvline(x=0.05, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].axhline(y=30.0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Smallest Eigenvalue')
    axes[1, 1].set_ylabel('Angle (degrees)')
    axes[1, 1].set_title('Singularity Detection Map')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Statistics summary
    axes[1, 2].axis('off')
    stats_text = f"""
    Statistics Summary:
    
    Total Points: {N}
    Modulation Points: {np.sum(modulation_flags)} ({100*np.sum(modulation_flags)/N:.1f}%)
    
    Manipulability Index:
    • Mean: {np.mean(manipulability_indices):.4f}
    • Std: {np.std(manipulability_indices):.4f}
    • Range: [{np.min(manipulability_indices):.4f}, {np.max(manipulability_indices):.4f}]
    
    Smallest Eigenvalue:
    • Mean: {np.mean(sigma_mins):.4f}
    • Std: {np.std(sigma_mins):.4f}
    • Below threshold: {np.sum(sigma_mins < 0.05)} ({100*np.sum(sigma_mins < 0.05)/N:.1f}%)
    
    Velocity-Eigenvector Angle:
    • Mean: {np.mean(angles_deg):.1f}°
    • Std: {np.std(angles_deg):.1f}°
    • Below threshold: {np.sum(angles_deg < 30.0)} ({100*np.sum(angles_deg < 30.0)/N:.1f}%)
    """
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig, axes


def main():

    # Initialize PyBullet
    cid, plane = setup_bullet(use_gui=True, timestep=1/240)
    
    # Load Panda and helper functions
    robot = load_panda()
    arm_joint_indices, qmin, qmax = get_joint_info(robot)
    ee_link = find_ee_link(robot)
    dof_joint_indices, dof_index_of_joint = get_dof_order(robot)
    arm_dof_cols = [dof_index_of_joint[j] for j in arm_joint_indices]

    # Learn and load DS
    se3_obj, p_init, q_init, p_att, q_att, p_test, q_test = learn_ds()

    # Plot p_test as waypoints
    draw_waypoints(np.asarray(p_test), point_color=(1, 0, 0), line_color=(1, 0, 0), point_size=5.0, line_width=2.0, lifeTime=0)

    # Move Panda to initial pose
    q_home = compute_ik(robot, ee_link, arm_joint_indices, dof_joint_indices, p_init, q_init)
    reset_arm(robot, arm_joint_indices, q_home)

    # Initialize list to store trajectory
    p_test = [p_init.reshape(1, -1)]
    q_test = [q_init]

    # Initialize lists to store manipulability data for analysis
    M_trajectory = []  # Store manipulability matrices
    velocities_trajectory = []  # Store velocities
    time_steps = []  # Store time steps
    modulation_flags_trajectory = []  # Store modulation flags

    # Run the DS in simulation
    tol = 0.03
    step_size = 0.01
    i = 0
    current_q = None  # Initialize outside loop
    while np.linalg.norm((q_test[-1] * q_att.inv()).as_rotvec()) >= tol or np.linalg.norm((p_test[-1] - p_att)) >= tol:
        
        p_in  = p_test[-1]
        q_in  = q_test[-1]

        p_next, q_next, gamma_pos, gamma_ori, v, w = se3_obj.step(p_in, q_in, step_size)

        modulation_flag, J = detect_singularity(robot, ee_link, dof_joint_indices, arm_dof_cols, v, i)

        # Store manipulability data for analysis
        # Compute manipulability matrix from Jacobian (use position part only for 3x3 matrix)
        J_pos = J[:3, arm_dof_cols]  # Position Jacobian (3x7)
        M = J_pos @ J_pos.T  # Manipulability matrix (3x3)
        M_trajectory.append(M)
        velocities_trajectory.append(v.flatten())
        time_steps.append(i * step_size)
        modulation_flags_trajectory.append(modulation_flag)

        if modulation_flag:
            X_m, Xdot_m = generate_modulated_samples(p_in.flatten(), v.flatten(), J, N=30)
            xdot = run_lmds_demo(p_in[0, :], X_m, Xdot_m, modulation_flag)
            p_next = integrate(p_in, xdot, dt=step_size).reshape(1, -1)

        p_test.append(p_next)
        q_test.append(q_next)
    
        q = compute_ik(robot, ee_link, arm_joint_indices, dof_joint_indices, 
                      p_next[0], q_next, current_q=current_q)
        set_arm_positions(robot, arm_joint_indices, q)
        current_q = q  
        
        p.stepSimulation()

        i += 1

    # Convert lists to numpy arrays for analysis
    M_trajectory = np.array(M_trajectory)
    velocities_trajectory = np.array(velocities_trajectory)
    time_steps = np.array(time_steps)
    modulation_flags_trajectory = np.array(modulation_flags_trajectory)

    # Analyze manipulability data using batch function
    modulation_flags, sigma_mins, angles_deg = detect_singularity_batch(
        M_trajectory, velocities_trajectory, sigma_threshold=0.05, angle_threshold=30.0)

    print(f"\nManipulability Analysis Results:")
    print(f"Total simulation steps: {len(M_trajectory)}")
    print(f"Steps requiring modulation: {np.sum(modulation_flags_trajectory)} ({100*np.sum(modulation_flags_trajectory)/len(modulation_flags_trajectory):.1f}%)")
    print(f"Steps with low manipulability: {np.sum(sigma_mins < 0.05)} ({100*np.sum(sigma_mins < 0.05)/len(sigma_mins):.1f}%)")

    # Plot manipulability analysis
    print("\nGenerating manipulability plots...")
    
    # Plot manipulability over time
    # plot_manipulability_over_time(M_trajectory, time_steps, 
    #                              title="Manipulability Analysis During Simulation")
    # plt.show()
    
    # Plot comprehensive manipulability statistics
    # plot_manipulability_statistics(M_trajectory, sigma_mins, angles_deg, modulation_flags,
                                #   title="Manipulability Statistics - Method1 Simulation")
    # plt.show()

    p.disconnect()
if __name__ == "__main__":
    main()
