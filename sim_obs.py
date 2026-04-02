import joblib
import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
from src.se3_lpvds.src.se3_class import se3_class
from src.se3_lpvds.src.quaternion_ds.src.quat_class import quat_class
from src.se3_lpvds.src.util import quat_tools
from scipy.spatial.transform import Rotation as R

def draw_frame(pos, orn, axis_len=0.1, line_width=3, life_time=0):
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

    x_axis = pos + axis_len * rot[:, 0]
    y_axis = pos + axis_len * rot[:, 1]
    z_axis = pos + axis_len * rot[:, 2]

    x_id = p.addUserDebugLine(pos, x_axis, [1, 0, 0], line_width, life_time, replaceItemUniqueId=x_id)  # X (red)
    y_id = p.addUserDebugLine(pos, y_axis, [0, 1, 0], line_width, life_time, replaceItemUniqueId=y_id)  # Y (green)
    z_id = p.addUserDebugLine(pos, z_axis, [0, 0, 1], line_width, life_time, replaceItemUniqueId=z_id)  # Z (blue)


# --- 1. SETUP & ENVIRONMENT ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=50, cameraPitch=-30, cameraTargetPosition=[0.5, 0, 0.4])
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# --- START RECORDING ---
# video_log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "obstacle_twist.mp4")

# Load Assets
p.loadURDF("plane.urdf")
robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
ee_idx = 11

# Create the Obstacle (Red Box)
obs_pos = [0.5, -0.18, 0.5]
visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                    halfExtents=[0.05, 0.05, 0.2],
                                    rgbaColor=[0.9, 0.1, 0.1, 1])
collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                          halfExtents=[0.05, 0.05, 0.2])
obstacleId = p.createMultiBody(baseMass=0,
                               baseCollisionShapeIndex=collisionShapeId,
                               baseVisualShapeIndex=visualShapeId,
                               basePosition=obs_pos)

# --- 2. LOAD MODELS ---
se3_obj   = se3_class.load("models/pouring-original_ds.pkl")
model_ori = se3_obj.ori_ds
model_pos = se3_obj.pos_ds
q_att     = se3_obj.q_att
gpr       = joblib.load("models/pouring_gpr.pkl")

steps = 500
t = np.linspace(0, 10, steps)

confidence_threshold = 2.0

# --- 3. EXECUTION LOOP ---
print("Running simulation...")
x_id, y_id, z_id = -1, -1, -1

# --- Original trajectory ---
pos_ori = se3_obj.p_in[0].copy()
orn_ori = model_ori.q_in[0].as_quat()

quat_history_ori = []
for i in range(steps):
    pos_ori, _, _ = model_pos._step(pos_ori.reshape(1, -1), dt=1./240.)
    pos_ori = pos_ori[0]

    orn_r, _, _ = model_ori._step(R.from_quat(orn_ori), step_size=se3_obj.dt)
    orn_ori = orn_r.as_quat()

    quat_history_ori.append(orn_ori.copy())
    time.sleep(1./240.)

print("Original trajectory done.")

# --- Modulated trajectory with decay + uncertainty ---
pos = se3_obj.p_in[0].copy()
orn = model_ori.q_in[0].as_quat()

quaternion_history = []
time_history = []

for i in range(steps):
    pos, _, _ = model_pos._step(pos.reshape(1, -1), dt=1./240.)
    pos = pos[0]

    q_curr = R.from_quat(orn)

    # GP prediction + uncertainty-based weight
    x_input = quat_tools.riem_log(q_att, q_curr).reshape(1, -1)
    y_pred, y_std = gpr.predict(x_input, return_std=True)

    weight = np.clip(1.0 - (np.mean(y_std) / confidence_threshold), 0.0, 1.0)
    y_override = y_pred * weight

    orn_r, _, _, _ = model_ori._step2(q_curr, se3_obj.dt, gpr, y_override=y_override)
    orn = orn_r.as_quat()

    quaternion_history.append(orn.copy())
    time_history.append(t[i])

    joint_poses = p.calculateInverseKinematics(robotId, ee_idx, pos, targetOrientation=orn)
    for j in range(7):
        p.resetJointState(robotId, j, joint_poses[j])

    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    axis_len = 0.12
    x_axis = pos + axis_len * rot[:, 0]
    y_axis = pos + axis_len * rot[:, 1]
    z_axis = pos + axis_len * rot[:, 2]

    x_id = p.addUserDebugLine(pos, x_axis, [1, 0, 0], lineWidth=3, lifeTime=0, replaceItemUniqueId=x_id)
    y_id = p.addUserDebugLine(pos, y_axis, [0, 1, 0], lineWidth=3, lifeTime=0, replaceItemUniqueId=y_id)
    z_id = p.addUserDebugLine(pos, z_axis, [0, 0, 1], lineWidth=3, lifeTime=0, replaceItemUniqueId=z_id)

    dist_to_obs = np.linalg.norm(np.array(pos) - np.array(obs_pos))
    if dist_to_obs < 0.35:
        p.addUserDebugLine(pos, obs_pos, [1, 0, 0], 2, 0.1)

    p.stepSimulation()
    time.sleep(1./100.)

# p.stopStateLogging(video_log_id)
print("Simulation complete.")
p.disconnect()

# --- PLOT QUATERNION ROLLOUT ---
quat_arr_ori = np.array(quat_history_ori)
quat_arr_mod = np.array(quaternion_history)
labels = ['x', 'y', 'z', 'w']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
    ax.plot(time_history, quat_arr_ori[:, i], '--', color=color, linewidth=1.5, alpha=0.6, label=f'Original q_{label}')
    ax.plot(time_history, quat_arr_mod[:, i], '-',  color=color, linewidth=2,              label=f'Modulated q_{label}')
    ax.set_ylabel(f'q_{label}')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('Time (s)')
plt.suptitle('Quaternion rollout: original vs modulated DS')
plt.tight_layout()
plt.show()
print("Quaternion plot displayed.")
