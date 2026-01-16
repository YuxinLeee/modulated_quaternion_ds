import pybullet as p
import pybullet_data
import time
import numpy as np

# --- 1. GENERATE DYNAMIC TRAJECTORY & ORIENTATION DATA ---
t = np.linspace(0, 10, 300)
event_center, event_width = 5.0, 1.2
gaussian = np.exp(-((t - event_center)**2) / (2 * event_width**2))

# Position Trajectory (Forward + Curve)
x_path = np.linspace(0.3, 0.7, len(t))
y_path = 0.2 * np.sin(t / 2)
z_path = 0.4 + 0.1 * np.cos(t / 2)

# Orientation (Euler: Roll, Pitch, Yaw)
base_roll, base_pitch, base_yaw = np.pi, 0, 0

# MODULATION
roll_mod = base_roll
pitch_mod = base_pitch + (0.35 * gaussian)
yaw_mod = base_yaw + (1.57 * gaussian)

# --- 2. SETUP PYBULLET ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=60, cameraPitch=-25, cameraTargetPosition=[0.5, 0, 0.4])
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# --- START RECORDING ---
video_log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "mani_guided.mp4")

p.loadURDF("plane.urdf")
robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# Initial Joint State
start_joints = [0, -0.78, 0, -2.356, 0, 1.57, 0.785]
for i in range(7):
    p.resetJointState(robotId, i, start_joints[i])

# Pre-draw the trajectory path in Red
for i in range(len(t)-1):
    p.addUserDebugLine([x_path[i], y_path[i], z_path[i]], 
                       [x_path[i+1], y_path[i+1], z_path[i+1]], 
                       [1, 0, 0], lineWidth=1, lifeTime=0)

# --- 3. EXECUTION LOOP ---
print("Simulating Motion + Orientation Modulation...")
ee_idx = 11

# Initialize IDs to -1 so PyBullet knows to create them on the first pass
x_id, y_id, z_id = -1, -1, -1

for i in range(len(t)):
    # 1. Target State
    pos = [x_path[i], y_path[i], z_path[i]]
    orn = p.getQuaternionFromEuler([roll_mod, pitch_mod[i], yaw_mod[i]])
    
    # 2. Inverse Kinematics
    joint_poses = p.calculateInverseKinematics(robotId, ee_idx, pos, targetOrientation=orn)
    
    # 3. Apply to Robot
    for j in range(7):
        p.resetJointState(robotId, j, joint_poses[j])
        
    # 4. Visualization (Smooth Updates)
    rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    l = 0.12  # Axis length
    
    # We update the lines using replaceItemUniqueId instead of creating new ones
    x_id = p.addUserDebugLine(pos, pos + rot_mat[:,0]*l, [1,0,0], lineWidth=3, lifeTime=0, replaceItemUniqueId=x_id)
    y_id = p.addUserDebugLine(pos, pos + rot_mat[:,1]*l, [0,1,0], lineWidth=3, lifeTime=0, replaceItemUniqueId=y_id)
    z_id = p.addUserDebugLine(pos, pos + rot_mat[:,2]*l, [0,0,1], lineWidth=3, lifeTime=0, replaceItemUniqueId=z_id)
    
    # Label the state
    if gaussian[i] > 0.5:
        p.addUserDebugText("MAX MODULATION", [0.5, 0, 0.8], [0, 0.8, 0], textSize=1.5, replaceItemUniqueId=1)
    else:
        p.addUserDebugText("FOLLOWING POLICY", [0.5, 0, 0.8], [0.5, 0.5, 0.5], textSize=1.2, replaceItemUniqueId=1)

    p.stepSimulation()
    time.sleep(1./240.)

p.stopStateLogging(video_log_id)
print("Simulation Complete.")