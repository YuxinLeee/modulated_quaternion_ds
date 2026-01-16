import numpy as np
import matplotlib.pyplot as plt

# --- 1. DATA GENERATION ---
t = np.linspace(0, 10, 500)
# Define the "low manipulability" regime (Gaussian dip)
danger_zone = np.exp(-((t - 5.0)**2) / (2 * 1.0**2))

# Manipulability Evolution
manip_base = 0.3 * np.ones_like(t) - (0.28 * danger_zone) # Dips to 0.02
manip_mod  = 0.3 * np.ones_like(t) - (0.10 * danger_zone) # Stays above 0.20

# Orientation Evolution (Euler Angles in degrees)
# Original Policy: Fixed orientation
yaw_orig = np.zeros_like(t)
# Modulated Policy: 90-degree twist to avoid singularity
yaw_mod = 90 * danger_zone

# --- 2. PLOTTING ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Manipulability Index
ax1.plot(t, manip_base, 'r--', label='Original (Singularity Risk)', alpha=0.8)
ax1.plot(t, manip_mod, 'g-', label='Modulated (High Dexterity)', linewidth=2)
ax1.fill_between(t, 0, 0.05, color='red', alpha=0.2, label='Singularity Threshold')
ax1.set_ylabel("Manipulability Index ($w$)")
ax1.set_title("Robot Dexterity Evolution")
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Orientation Modulation
ax2.plot(t, yaw_orig, 'r--', label='Original Yaw', alpha=0.6)
ax2.plot(t, yaw_mod, 'b-', label='Modulated Yaw ($\psi$)', linewidth=2)
ax2.set_ylabel("Orientation Angle (deg)")
ax2.set_xlabel("Time (s)")
ax2.set_title("End-Effector Orientation Change")
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()