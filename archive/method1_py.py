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
    
    if pos_error > 0.01 or orn_error > 0.1:
        print(f"Warning: Large IK error - pos: {pos_error:.4f}, orn: {orn_error:.4f}")
        
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
    # plot_tools.plot_result(p_in, p_test, q_test)

    # plot_tools.plot_gamma(gamma_pos.T, title="pos")
    # plot_tools.plot_gamma(gamma_ori.T, title="ori")

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


    # Run the DS in simulation
    tol = 0.01
    step_size = 0.01
    i = 0
    current_q = None  # Initialize outside loop
    while np.linalg.norm((q_test[-1] * q_att.inv()).as_rotvec()) >= tol or np.linalg.norm((p_test[-1] - p_att)) >= tol:
        
        p_in  = p_test[-1]
        q_in  = q_test[-1]

        p_next, q_next, gamma_pos, gamma_ori, v, w = se3_obj.step(p_in, q_in, step_size)

        modulation_flag, J = detect_singularity(robot, ee_link, dof_joint_indices, arm_dof_cols, v, i)

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

    p.disconnect()
if __name__ == "__main__":
    main()
