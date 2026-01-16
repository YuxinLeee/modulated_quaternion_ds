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
from src.se3_lpvds.src.quaternion_ds.src.quat_class import quat_class

from gp_modulate import GaussianProcessModulatedDS, LPVDSWrapper
from sample import sample_around_trajectory, plot_samples
import scipy.io as sio
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
    import pybullet as p
    import pybullet_data

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
    import pybullet as p

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
    import pybullet as p

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

def get_dof_order(robot: int) -> Tuple[List[int], dict]:
    import pybullet as p

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
    import pybullet as p

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
    
    if pos_error > 0.01 or orn_error > 0.1:
        print(f"Warning: Large IK error - pos: {pos_error:.4f}, orn: {orn_error:.4f}")
        
    return q



def learn_ds(p_in, q_in, p_out, q_out, p_att, q_att, dt, p_init, q_init):
    '''Run lpvds'''
    se3_obj = se3_class(p_in, q_in, p_out, q_out, p_att, q_att, dt, K_init=4)
    se3_obj.begin()

    
    '''Evaluate results'''
    p_init = p_init[0]
    q_init = R.from_quat(-q_init[0].as_quat())
    p_test, q_test, gamma_pos, gamma_ori, v_test, w_test = se3_obj.sim(p_init, q_init, step_size=0.05)

    '''Plot results'''
    plot_tools.plot_result(p_in, p_test, q_test)

    # plot_tools.plot_gamma(gamma_pos.T, title="pos")
    # plot_tools.plot_gamma(gamma_ori.T, title="ori")

    plt.show()

    return se3_obj #, p_init, q_init, p_att, q_att, p_test, q_test




def compute_trajectory_jacobians(robot: int, ee_link: int, p_in: np.ndarray, q_in: List[R], 
                               arm_joint_indices: List[int], dof_joint_indices: List[int]) -> List[np.ndarray]:
    """Compute Jacobians for all points in the trajectory.
    
    Args:
        robot: PyBullet robot ID
        ee_link: End-effector link index
        p_in: Position trajectory (N, 3)
        q_in: List of N orientation rotations
        arm_joint_indices: Indices of arm joints
        dof_joint_indices: Indices of DoF joints
    
    Returns:
        List[np.ndarray]: List of Jacobian matrices for each point
    """
    jacobians = []
    
    for i in range(len(p_in)):
        # Compute IK for this pose
        target_pos = p_in[i]
        target_quat = q_in[i].as_quat()
        
        # Get joint angles for this pose
        joint_poses = p.calculateInverseKinematics(
            robot, ee_link, 
            target_pos.tolist(),
            target_quat.tolist()
        )
        
        # Set robot to this configuration
        for j, joint_idx in enumerate(arm_joint_indices):
            p.resetJointState(robot, joint_idx, joint_poses[j])
            
        # Get current joint positions for Jacobian calculation
        q_dof = [p.getJointState(robot, j)[0] for j in dof_joint_indices]
        
        # Calculate Jacobian
        J = calculate_jacobian(robot, ee_link, q_dof)
        jacobians.append(J[:3, :]) # only the posirion component of the Jacobian
        
    return jacobians



def call_gmr_mani_3d(positions: np.ndarray, matlab_path: str = None, model_path: str = None) -> np.ndarray:
    """
    Call the MATLAB GMR_mani_3d.m function to get manipulability at 3D position(s).
    
    Args:
        positions: 3D position(s) as numpy array. Can be:
                  - Single position: [x, y, z] shape (3,)
                  - Multiple positions: [[x1, y1, z1], [x2, y2, z2], ...] shape (N, 3)
        matlab_path: Path to the directory containing GMR_mani_3d.m
        model_path: Path to the directory containing modelPD.mat (if different from matlab_path)
    
    Returns:
        np.ndarray: Manipulability matrices. Shape:
                   - Single position: (3, 3) matrix
                   - Multiple positions: (N, 3, 3) array of matrices
    """
    if matlab_path is None:
        matlab_path = '/Users/macpro/Desktop/Project/Manipulability-master/fcts'
    
    if model_path is None:
        model_path = '/Users/macpro/Desktop/Project/mani_py'
    
    # Handle single position input
    if positions.ndim == 1:
        positions = positions.reshape(1, -1)
    
    n_positions = positions.shape[0]
    
    # Initialize output array to store 3x3 matrices
    manipulability_matrices = np.zeros((n_positions, 3, 3))
    
    import matlab.engine
    eng = matlab.engine.start_matlab()
    
    # Add the necessary paths
    eng.addpath(matlab_path, nargout=0)
    eng.addpath(model_path, nargout=0)  # Add path where modelPD.mat is located
    
    # Change to the directory containing modelPD.mat
    eng.cd(model_path, nargout=0)
    
    print(f"Computing manipulability for {n_positions} position(s)...")
    
    # Process each position
    for i, pos in enumerate(positions):
        # Convert position to MATLAB format (3x1 column vector)
        matlab_position = matlab.double(pos.reshape(-1, 1).tolist())
        
        # Call GMR_mani_3d function
        result = eng.GMR_mani_3d(matlab_position, nargout=1)
        
        # Convert result back to numpy array
        M = np.array(result)
        
        # Convert from vector form to 3x3 symmetric matrix
        # The MATLAB function vec2symmat converts vector to symmetric matrix
        # For a 3x3 matrix, the vector form should have 6 elements: [M11, M12, M22, M13, M23, M33]# M33
        manipulability_matrices[i] = M

    # Stop MATLAB engine
    eng.quit()
    
    print(f"Successfully computed manipulability for {n_positions} position(s)")
    
    # Return single matrix if input was single position, otherwise return array
    if n_positions == 1:
        return manipulability_matrices[0]
    else:
        return manipulability_matrices
        


def call_mani_learn(jacobians: List[np.ndarray], positions: np.ndarray, 
                   matlab_path: str = None) -> str:
    """
    Call the MATLAB mani_learn.m function with trajectory data using MATLAB Engine.
    
    Args:
        jacobians: List of Jacobian matrices for each trajectory point
        positions: Position trajectory (N, 3)
        matlab_path: Path to the Manipulability-master directory containing mani_learn.m
    
    """
    if matlab_path is None:
        matlab_path = '/Users/macpro/Desktop/Project/Manipulability-master/examples/Learning'
    
    # Convert jacobians list to 3D numpy array
    # jacobians is a list of (6, n_joints) arrays
    n_points = len(jacobians)
    n_rows, n_cols = jacobians[0].shape
    jacobian_array = np.zeros((n_rows, n_cols, n_points))
    
    for i, jac in enumerate(jacobians):
        jacobian_array[:, :, i] = jac
    
    import matlab.engine
    eng = matlab.engine.start_matlab()
    
    # Add the Manipulability-master path
    eng.addpath(matlab_path, nargout=0)
    eng.addpath(os.path.join(matlab_path, '..',  '..', 'fcts'), nargout=0)
    
    # Convert numpy arrays to MATLAB arrays and pass directly
    matlab_jacobian = matlab.double(jacobian_array.tolist())
    matlab_positions = matlab.double(positions.T.tolist())  # MATLAB expects 3xN format
    
    # Call mani_learn function directly with the data
    eng.mani_learn(matlab_jacobian, matlab_positions, nargout=0)
    
    # Stop MATLAB engine
    eng.quit()
    
    print("Successfully called mani_learn using MATLAB engine")
        


def detect_singularity(M_list, velocities, sigma_threshold=0.05, angle_threshold=30.0):
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


def generate_modulated_samples(x: np.ndarray, x_dot: np.ndarray, M: np.ndarray) -> tuple:
    """Generate modulated velocities for given positions using velocity and manipulability matrix.
    
    Args:
        x: Positions to use as X_m. Can be:
           - Single position: (3,) or (3, 1) 
           - Multiple positions: (N, 3) or (3, N)
        x_dot: Current velocity (3,) or (3, 1)  
        M: Manipulability matrix (3, 3)
        
    Returns:
        tuple: (X_m, Xdot_m) where X_m is the input positions and Xdot_m is modulated velocities
    """
    # Step 1: Use provided positions as X_m
    # Handle different input shapes
    if x.ndim == 1:
        # Single position (3,)
        X_m = x.reshape(3, 1)
        N = 1
    else:
        # Positions in format (N, 3)
        X_m = x.T
        N = x.shape[0]
    
    # Step 2: Get velocity magnitude
    x_dot_norm = np.linalg.norm(x_dot, axis=0)
    
    # Step 3: Get principal direction from manipulability matrix M
    # The manipulability ellipsoid M = J*J^T, so we can get principal directions from M
    eigenvals, eigenvecs = np.linalg.eigh(M)
    
    # Find index of largest eigenvalue using argmax
    max_idx = np.argmax(eigenvals, axis=1)
    principal_dir = eigenvecs[np.arange(N), :, max_idx]  # Shape: (M, 3)
    
    # Ensure consistent direction by aligning with current velocity
    if x_dot.ndim == 1:
        x_dot = np.tile(x_dot.reshape(-1, 1), (1, N))  # Shape: (3, M)
    
    # Normalize velocities
    x_dot_norms = np.linalg.norm(x_dot, axis=0)  # Shape: (M,)
    valid_mask = x_dot_norms > 0
    
    if np.any(valid_mask):
        x_dot_normalized = x_dot / (x_dot_norms + 1e-8)  # Shape: (3, M)
        
        # Check alignment for each principal direction with corresponding velocity
        # principal_dir shape: (M, 3), x_dot_normalized shape: (3, M)
        dot_products = np.sum(principal_dir.T * x_dot_normalized, axis=0)  # Shape: (M,)
        
        # Flip principal directions that point opposite to velocity
        flip_mask = dot_products < 0
        principal_dir[flip_mask] = -principal_dir[flip_mask]
    
    # Step 4: Compute modulated velocities
    # principal_dir shape: (M, 3), x_dot_norms shape: (M,)
    Xdot_m = (principal_dir * x_dot_norms.reshape(-1, 1)).T  # Shape: (3, M) 
    
    return X_m, Xdot_m


def reset_arm(robot: int, arm_joint_indices: List[int], q_arm: np.ndarray):
    import pybullet as p

    for idx, q in zip(arm_joint_indices, q_arm):
        p.resetJointState(robot, idx, float(q))


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


def plot_trajectories_with_arrows(p_in, p_out, X_m, Xdot_m, M_list=None, title="Trajectory Comparison", 
                                 arrow_scale=0.1, ellipsoid_scale=0.05, figsize=(12, 8)):
    """
    Plot p_in as points with p_out as arrows, and X_m with Xdot_m as arrows in different colors.
    Optionally draw ellipsoids at X_m positions based on manipulability matrices.
    
    Args:
        p_in: Input positions, shape (N, 3) or (3, N)
        p_out: Output velocities for p_in, shape (N, 3) or (3, N) 
        X_m: Modulated positions, shape (N, 3) or (3, N)
        Xdot_m: Modulated velocities, shape (N, 3) or (3, N)
        M_list: Manipulability matrices, shape (N, 3, 3), optional
        title: Plot title
        arrow_scale: Scale factor for arrow lengths
        ellipsoid_scale: Scale factor for ellipsoid sizes
        figsize: Figure size tuple
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Ensure consistent shape (N, 3) for all arrays
    def ensure_shape(arr):
        if arr.shape[1] == 3:
            return arr  # Already (N, 3)
        else:
            return arr.T  # Convert from (3, N) to (N, 3)
    
    p_in = ensure_shape(p_in)
    p_out = ensure_shape(p_out)
    X_m = ensure_shape(X_m)
    Xdot_m = ensure_shape(Xdot_m)
    
    # Plot original trajectory (p_in, p_out)
    ax.scatter(p_in[:, 0], p_in[:, 1], p_in[:, 2], 
              c='blue', s=30, alpha=0.7, label='Original Points (p_in)')
    
    # Add arrows for original trajectory
    ax.quiver(p_in[:, 0], p_in[:, 1], p_in[:, 2],
              p_out[:, 0] * arrow_scale, p_out[:, 1] * arrow_scale, p_out[:, 2] * arrow_scale,
              color='blue', alpha=0.6, arrow_length_ratio=0.1, linewidth=1)
    
    # Plot modulated trajectory (X_m, Xdot_m)
    ax.scatter(X_m[:, 0], X_m[:, 1], X_m[:, 2], 
              c='red', s=30, alpha=0.7, label='Modulated Points (X_m)')
    
    # Add arrows for modulated trajectory
    ax.quiver(X_m[:, 0], X_m[:, 1], X_m[:, 2],
              Xdot_m[:, 0] * arrow_scale, Xdot_m[:, 1] * arrow_scale, Xdot_m[:, 2] * arrow_scale,
              color='red', alpha=0.6, arrow_length_ratio=0.1, linewidth=1)
    
    # Add ellipsoids at X_m positions if M_list is provided
    if M_list is not None:
        for i in range(X_m.shape[0]):
            center = X_m[i, :]
            M = M_list[i]  # Shape: (3, 3)
            
            # Create ellipsoid from manipulability matrix
            # The ellipsoid is defined by the eigenvalues and eigenvectors of M
            eigenvals, eigenvecs = np.linalg.eigh(M)
            
            # Ensure positive eigenvalues for ellipsoid
            eigenvals = np.abs(eigenvals)
            
            # Scale the ellipsoid
            radii = np.sqrt(eigenvals) * ellipsoid_scale
            
            # Generate ellipsoid surface
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Scale by eigenvalues (radii)
            x_ellipsoid = x_sphere * radii[0]
            y_ellipsoid = y_sphere * radii[1]
            z_ellipsoid = z_sphere * radii[2]
            
            # Rotate by eigenvectors
            for j in range(x_ellipsoid.shape[0]):
                for k in range(x_ellipsoid.shape[1]):
                    point = np.array([x_ellipsoid[j, k], y_ellipsoid[j, k], z_ellipsoid[j, k]])
                    rotated_point = eigenvecs @ point
                    x_ellipsoid[j, k] = rotated_point[0]
                    y_ellipsoid[j, k] = rotated_point[1]
                    z_ellipsoid[j, k] = rotated_point[2]
            
            # Translate to center position
            x_ellipsoid += center[0]
            y_ellipsoid += center[1]
            z_ellipsoid += center[2]
            
            # Plot ellipsoid surface
            ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, 
                           alpha=0.3, color='green', linewidth=0.5, edgecolor='darkgreen')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([
        np.concatenate([p_in[:, 0], X_m[:, 0]]).max() - np.concatenate([p_in[:, 0], X_m[:, 0]]).min(),
        np.concatenate([p_in[:, 1], X_m[:, 1]]).max() - np.concatenate([p_in[:, 1], X_m[:, 1]]).min(),
        np.concatenate([p_in[:, 2], X_m[:, 2]]).max() - np.concatenate([p_in[:, 2], X_m[:, 2]]).min()
    ]).max() / 2.0
    
    mid_x = (np.concatenate([p_in[:, 0], X_m[:, 0]]).max() + np.concatenate([p_in[:, 0], X_m[:, 0]]).min()) * 0.5
    mid_y = (np.concatenate([p_in[:, 1], X_m[:, 1]]).max() + np.concatenate([p_in[:, 1], X_m[:, 1]]).min()) * 0.5
    mid_z = (np.concatenate([p_in[:, 2], X_m[:, 2]]).max() + np.concatenate([p_in[:, 2], X_m[:, 2]]).min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig, ax


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
    axes[0, 2].axvline(60.0, color='red', linestyle='--', label='Threshold: 60°')
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
    axes[1, 1].axhline(y=60.0, color='red', linestyle='--', alpha=0.7)
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
    • Below threshold: {np.sum(angles_deg < 60.0)} ({100*np.sum(angles_deg < 60.0)/N:.1f}%)
    """
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig, axes


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


def integrate(x, xdot, dt=0.01):
        return x + xdot * dt
    

def main():
    """Setup bullet and load robot"""
    cid, plane = setup_bullet(use_gui=True, timestep=1/240)
    robot = load_panda()
    arm_joint_indices, qmin, qmax = get_joint_info(robot)
    ee_link = find_ee_link(robot)
    dof_joint_indices, dof_index_of_joint = get_dof_order(robot)
    arm_dof_cols = [dof_index_of_joint[j] for j in arm_joint_indices]


    """Load trajectory data"""
    folder_path = '/Users/macpro/Desktop/Project/infeasible_task2_0214/all.mat'
    p_raw, q_raw, t_raw, dt = load_tools._process_bag(folder_path, if_flip=True)
    p_in, q_in, t_in             = process_tools.pre_process(p_raw, q_raw, t_raw,  shift=True, opt= "savgol")
    p_out, q_out                 = process_tools.compute_output(p_in, q_in, t_in)
    p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)
    p_in, q_in, p_out, q_out     = process_tools.rollout_list(p_in, q_in, p_out, q_out)


    """Create LPVDS wrapper and GP-MDS"""
    lpvds_ds = LPVDSWrapper(p_in, p_out, p_att)
    p_test = lpvds_ds.lpvds.sim(p_init[0].reshape(1, -1), dt)
    gp_mds = GaussianProcessModulatedDS(lpvds_ds)
    gp_mds.gpr.set_hyperparams(0.05, 1.0, 0.02)

    quat_ds = quat_class(q_in, q_out, q_att, dt, K_init=3)
    quat_ds.begin()


    """Compute Jacobians for all trajectory points"""
    jacobians = compute_trajectory_jacobians(
        robot, ee_link, p_in, q_in,
        arm_joint_indices, dof_joint_indices
    )

    
    """Calling MATLAB mani_learn function"""
    call_mani_learn(jacobians, p_in)
    

    """Generate modulated data"""
    samples_lhs, mask_lhs, bounds = sample_around_trajectory(
        p_in,
        margin=0.2,
        N=100,
        method='lhs', # 'halton' or 'lhs',
        distance_threshold=0.05,
        rng_seed=7
    )
    # plot_samples(p_in, samples_lhs, mask_lhs, title="3D Arch with LHS Coverage", azim=35)

    sampled_pts = samples_lhs[mask_lhs]
    
    M_list    = call_gmr_mani_3d(sampled_pts)
    xdot_list = lpvds_ds.lpvds.predict(sampled_pts)
    modulation_flags, sigma_mins, angles_deg = detect_singularity(M_list, xdot_list, sigma_threshold=0.05, angle_threshold=60.0)
    X_m, Xdot_m = generate_modulated_samples(sampled_pts[modulation_flags], xdot_list[:, modulation_flags], M_list[modulation_flags])



    # """Plot trajectories comparison"""
    # plot_trajectories_with_arrows(p_in, p_out, X_m.T, Xdot_m.T, M_list[modulation_flags],
    #                              title="Original vs Modulated Trajectories with Manipulability Ellipsoids", 
    #                              arrow_scale=0.15, ellipsoid_scale=0.02)
    # plt.show()

    # """Plot manipulability analysis"""
    # # Plot manipulability over sampled points (treating as time series)
    # plot_manipulability_over_time(M_list, title="Manipulability Analysis Over Sampled Points")
    # plt.show()
    
    # # Plot comprehensive manipulability statistics
    # plot_manipulability_statistics(M_list, sigma_mins, angles_deg, modulation_flags)
    # plt.show()

    """Add modulated data to GP-MDS"""
    gp_mds.add_data_batch(X_m, Xdot_m)



    """Simulate the modulated system"""
    # Move Panda to initial pose
    q_home = compute_ik(robot, ee_link, arm_joint_indices, dof_joint_indices, p_init[0], q_init[0])
    reset_arm(robot, arm_joint_indices, q_home)
    draw_waypoints(p_test, point_color=(1, 0, 0), line_color=(1, 0, 0), point_size=5.0, line_width=2.0, lifeTime=0)

    # Initialize list to store trajectory
    p_test = [p_init[0].reshape(1, -1)]
    q_test = [q_init[0]]

    # Run the DS in simulation
    tol = 0.01
    step_size = 0.01
    i = 0
    current_q = None  # Initialize outside loop

    while np.linalg.norm((q_test[-1] * q_att.inv()).as_rotvec()) >= tol or np.linalg.norm((p_test[-1] - p_att)) >= tol:
        
        p_in  = p_test[-1]
        q_in  = q_test[-1]

        q_next, gamma_ori, w = quat_ds._step(q_in, step_size)

        # p_next, gamma_pos, v = lpvds_ds.lpvds._step(p_in, step_size)
        xdot = gp_mds.get_output(p_in[0])
        p_next = integrate(p_in, xdot, dt=step_size).reshape(1, -1)

        p_test.append(p_next)
        q_test.append(q_next)
    
        q = compute_ik(robot, ee_link, arm_joint_indices, dof_joint_indices, 
                      p_next[0], q_next, current_q=current_q)
        set_arm_positions(robot, arm_joint_indices, q)
        current_q = q  
        
        p.stepSimulation()

        i += 1
        print(f"Iteration: {i}")
    p.disconnect()




if __name__ == "__main__":
    main()

