import pybullet as p
import pybullet_data
import numpy as np
from typing import List, Tuple
from src.se3_lpvds.src.se3_class import se3_class
from src.se3_lpvds.src.util import load_tools, process_tools
from scipy.spatial.transform import Rotation as R
from src.se3_lpvds.src.util.load_tools import load_single_h5_UMI

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

def get_full_joint_vector(robot: int, dof_joint_indices: List[int]) -> List[float]:
    q = []
    for j in dof_joint_indices:
        qj = p.getJointState(robot, j)[0]
        q.append(qj)
    return q

def set_arm_positions(robot: int, arm_joint_indices: List[int], q_arm: np.ndarray, kp=1.0, kd=0.1):

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

    for idx, q in zip(arm_joint_indices, q_arm):
        p.resetJointState(robot, idx, float(q))



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
    
    # actual_pos = np.array(ee_state[4])
    # actual_orn = np.array(ee_state[5])
    
    # pos_error = np.linalg.norm(actual_pos - p_target)
    # orn_error = np.linalg.norm(actual_orn - q_target.as_quat())
    
    # if pos_error > 0.01 or orn_error > 0.1:
    #     print(f"Warning: Large IK error - pos: {pos_error:.4f}, orn: {orn_error:.4f}")
        
    return q


def draw_waypoints(points: np.ndarray, point_color=(0.9, 0.3, 0.1), line_color=(1, 0, 0), point_size: float = 5.0, line_width: float = 2.0, lifeTime: float = 0.0):
    """Draw waypoints as debug points plus connecting lines in PyBullet.
    - points: (N, 3) array-like in world coordinates
    - point_color/line_color: RGB in [0,1]
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    p.addUserDebugPoints(pointPositions=pts.tolist(), pointColorsRGB=[point_color] * len(pts), pointSize=point_size, lifeTime=lifeTime)



def manipulability_with_sigma_min(J: np.ndarray, joint_cols: List[int]) -> Tuple[float, float]:
    """Return (Yoshikawa manipulability, smallest singular value)."""
    J7 = J[:, joint_cols]
    U, s, _ = np.linalg.svd(J7, full_matrices=False)
    m = float(np.prod(s))
    sigma_min = float(np.min(s)) if s.size > 0 else float("nan")
    return m, sigma_min

def smallest_singular_vector_linear(J: np.ndarray, joint_cols: List[int]) -> Tuple[np.ndarray, float]:
    """Return unit left singular vector of Jv with smallest singular value and sigma_min.

    Jv is the 3xN linear part of the Jacobian. Returns (u_min, sigma_min).
    """
    Jv = J[3:, joint_cols]
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


def smallest_singular_vector_angular(J: np.ndarray, joint_cols: List[int]) -> Tuple[np.ndarray, float]:
    """Return unit left singular vector of Jv with smallest singular value and sigma_min.

    Jv is the 3xN linear part of the Jacobian. Returns (u_min, sigma_min).
    """
    Jv = J[3:, joint_cols]
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


def calculate_angle_to_svec(vel: np.ndarray, u_min: np.ndarray) -> float:
    """Calculate angle between velocity vector and smallest singular vector.
    
    Args:
        v: Velocity vector (3,)
        svec_min: Smallest singular vector (3,)
    
    Returns:
        float: Angle in degrees, or nan if velocity is zero
    """
    vel = vel[:,0] if vel.ndim > 1 else vel
    v_normalized = vel / np.linalg.norm(vel)
    dot_product = np.dot(v_normalized, u_min)
    dot_product = np.clip(abs(dot_product), -1.0, 1.0)  # Ensure valid arccos input
    return np.degrees(np.arccos(dot_product))

    

def detect_singularity(robot, ee_link, dof_joint_indices, arm_dof_cols, curr_ang_vel, i):
    q_dof = get_full_joint_vector(robot, dof_joint_indices)
    J = calculate_jacobian(robot, ee_link, q_dof)
    m, sigma_min = manipulability_with_sigma_min(J, arm_dof_cols)
    # u_min, sigma_min = smallest_singular_vector_linear(J, arm_dof_cols)
    u_min, sigma_min = smallest_singular_vector_angular(J, arm_dof_cols)
    angle_deg = calculate_angle_to_svec(curr_ang_vel, u_min)
    modulation_flag = True if sigma_min < 0.05 and angle_deg < 30.0 else False

    print(
        f"step {i}: sigma_min {sigma_min:.6e}, "
        f"angle: {angle_deg:.2f}°"
        f" -> modulation: {modulation_flag}"
    )

    # return modulation_flag, J



def run_sim(task_name):
    
    """Initialize PyBullet"""
    cid, plane = setup_bullet(use_gui=True, timestep=1/240)
    

    """Load Panda and helper functions"""
    robot = load_panda()
    arm_joint_indices, qmin, qmax = get_joint_info(robot)
    ee_link = find_ee_link(robot)
    dof_joint_indices, dof_index_of_joint = get_dof_order(robot)
    arm_dof_cols = [dof_index_of_joint[j] for j in arm_joint_indices]


    """Learn and load DS"""
    se3_obj = se3_class.load("models/" + task_name + "_ds.pkl")
    p_0 = se3_obj.p_in[0]
    q_0 = se3_obj.q_in[0]
    p_att = se3_obj.p_att
    q_att = se3_obj.q_att
    dt = se3_obj.dt
    T = dt * len(se3_obj.q_in)
    p_test_list = []
    q_0 = R.from_quat(q_0.as_quat())
    p_test, q_test, gamma_pos, gamma_ori, v_test, w_test = se3_obj.sim(p_0, q_0, p_att, q_att, step_size=dt, duration=T)
    p_test_list.append(p_test)
    draw_waypoints(np.asarray(p_test_list[0]), point_color=(1, 0, 0), line_color=(1, 0, 0), point_size=5.0, line_width=2.0, lifeTime=0)


    """Move Panda to initial pose"""
    q_home = compute_ik(robot, ee_link, arm_joint_indices, dof_joint_indices, p_0, q_0)
    reset_arm(robot, arm_joint_indices, q_home)


    """Run the DS in simulation"""
    p_test = [p_0.reshape(1, -1)]
    q_test = [q_0]
    tol = 0.03
    i = 0
    current_q = None  # Initialize outside loop
    while np.linalg.norm((q_test[-1] * q_att.inv()).as_rotvec()) >= tol or np.linalg.norm((p_test[-1] - p_att)) >= tol:
        
        p_in  = p_test[-1]
        q_in  = q_test[-1]

        p_next, q_next, _, _, v, w = se3_obj._step(p_in, q_in, se3_obj.dt)

        detect_singularity(robot, ee_link, dof_joint_indices, arm_dof_cols, w, i)

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
    run_sim("test_oculus")