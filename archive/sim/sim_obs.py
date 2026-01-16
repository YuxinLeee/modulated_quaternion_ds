import pybullet as p
import pybullet_data
import time
import numpy as np


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

# Set camera BEFORE recording starts for a clean shot
p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=50, cameraPitch=-30, cameraTargetPosition=[0.5, 0, 0.4])
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Turn off sidebars for cleaner video

# --- START RECORDING ---
# Note: Requires FFmpeg installed on system
video_log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "obstacle_twist.mp4")

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

# --- 2. TRAJECTORY GENERATION ---
# Reduced steps for faster, snappier motion (300 instead of 400)
steps = 300 
t = np.linspace(0, 10, steps)

# Spatial Path
x_path = np.linspace(0.3, 0.7, steps)
y_path = np.zeros(steps)
z_path = np.ones(steps) * 0.5

# Modulation Logic
modulation_strength = np.exp(-((t - 5.0)**2) / (2 * 0.8**2))

# Orientation Planning
base_roll = np.pi
base_pitch = 0
base_yaw = 0
twist_magnitude = np.pi / 2 

yaw_profile = base_yaw + (twist_magnitude * modulation_strength)
roll_profile = base_roll + (0.3 * modulation_strength)

# --- 3. EXECUTION LOOP ---
print("Running simulation... (Recording to obstacle_twist.mp4)")
x_id, y_id, z_id = -1, -1, -1

for i in range(steps):
    pos = [x_path[i], y_path[i], z_path[i]]
    orn = p.getQuaternionFromEuler([roll_profile[i], base_pitch, yaw_profile[i]])
    
    joint_poses = p.calculateInverseKinematics(robotId, ee_idx, pos, targetOrientation=orn)
    for j in range(7):
        p.resetJointState(robotId, j, joint_poses[j])
    
    # Visual Cues
    # p.addUserDebugLine([x_path[i], y_path[i], z_path[i]], [x_path[i], y_path[i], 0], [0,0,0], 1, 0.1)
    # draw_frame(pos, orn)
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    axis_len = 0.12
    x_axis = pos + axis_len * rot[:, 0]
    y_axis = pos + axis_len * rot[:, 1]
    z_axis = pos + axis_len * rot[:, 2]

    x_id = p.addUserDebugLine(pos, x_axis, [1, 0, 0], lineWidth=3, lifeTime=0, replaceItemUniqueId=x_id)  # X (red)
    y_id = p.addUserDebugLine(pos, y_axis, [0, 1, 0], lineWidth=3, lifeTime=0, replaceItemUniqueId=y_id)  # Y (green)
    z_id = p.addUserDebugLine(pos, z_axis, [0, 0, 1], lineWidth=3, lifeTime=0, replaceItemUniqueId=z_id)  # Z (blue)

    dist_to_obs = np.linalg.norm(np.array(pos) - np.array(obs_pos))
    if dist_to_obs < 0.35:
        p.addUserDebugLine(pos, obs_pos, [1, 0, 0], 2, 0.1)
    
    if modulation_strength[i] > 0.1:
        status = f"Modulating Orientation"
        color = [1, 0, 0]
    else:
        status = "NOMINAL PATH"
        color = [0, 0, 1]
        
    p.addUserDebugText(status, [0.4, 0.3, 0.7], color, textSize=1.5, replaceItemUniqueId=1)

    p.stepSimulation()
    
    # FASTER PLAYBACK: Reduced sleep time significantly
    time.sleep(1./240.) 

# --- STOP RECORDING ---
p.stopStateLogging(video_log_id)
print("Simulation complete. Video saved.")
p.disconnect()