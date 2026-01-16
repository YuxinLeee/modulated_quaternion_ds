import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def lhs_sample(N, bounds, rng=None):
    """Latin Hypercube in d-D with jitter & independent column permutations."""
    if rng is None:
        rng = np.random.default_rng()
    d = bounds.shape[0]
    U = (rng.random((N, d)) + np.arange(N)[:, None]) / N
    for j in range(d):
        rng.shuffle(U[:, j])
    return bounds[:, 0] + U * (bounds[:, 1] - bounds[:, 0])

def van_der_corput(n, base=2):
    seq = np.zeros(n)
    for i in range(n):
        x = 0.0
        denom = 1.0
        k = i + 1  # start from 1 to avoid leading zeros issue
        while k > 0:
            k, r = divmod(k, base)
            denom *= base
            x += r / denom
        seq[i] = x
    return seq

def halton_sample(N, bounds, bases=(2, 3, 5), skip=64):
    """Basic (unscrambled) Halton in d-D with initial skip."""
    d = bounds.shape[0]
    n_gen = N + skip
    H = np.zeros((N, d))
    for j in range(d):
        h = van_der_corput(n_gen, base=bases[j])[skip:][:N]
        H[:, j] = h
    return bounds[:, 0] + H * (bounds[:, 1] - bounds[:, 0])

def distances_to_curve(P: np.ndarray, trajectory: np.ndarray, n_grid: int = 4000) -> tuple:
    """Calculate closest Euclidean distance from points P to trajectory.
    
    Args:
        P: Points to check, shape (N, 3)
        trajectory: Reference trajectory, shape (T, 3)
        n_grid: Number of interpolation points
    
    Returns:
        tuple: (dmin, tstar)
            - dmin: Minimum distances, shape (N,)
            - tstar: Parameter values at minimum distances
    """
    # Interpolate trajectory to regular grid
    T = trajectory.shape[0]
    t_orig = np.linspace(0, 1, T)
    t_grid = np.linspace(0, 1, n_grid)
    
    # Interpolate each coordinate
    Cx = np.interp(t_grid, t_orig, trajectory[:, 0])
    Cy = np.interp(t_grid, t_orig, trajectory[:, 1])
    Cz = np.interp(t_grid, t_orig, trajectory[:, 2])
    
    N = P.shape[0]
    dmin = np.empty(N)
    tstar = np.empty(N)
    
    # Process in chunks to avoid memory issues
    chunk = 500
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        Px = P[i0:i1, 0][:, None]
        Py = P[i0:i1, 1][:, None]
        Pz = P[i0:i1, 2][:, None]
        
        # Compute squared distances to all grid points
        D2 = (Px - Cx[None, :])**2 + (Py - Cy[None, :])**2 + (Pz - Cz[None, :])**2
        
        # Find minimum distances and corresponding parameters
        jmin = np.argmin(D2, axis=1)
        dmin[i0:i1] = np.sqrt(D2[np.arange(i1 - i0), jmin])
        tstar[i0:i1] = t_grid[jmin]
        
    return dmin, tstar


def sample_around_trajectory(trajectory: np.ndarray, 
                           margin: float = 0.2,
                           N: int = 800,
                           method: str = 'lhs',
                           distance_threshold: float = 0.2,
                           rng_seed: int = None) -> tuple:
    """
    Sample points around a 3D trajectory and return points within specified distance.
    
    Args:
        trajectory: Array of shape (T, 3) containing trajectory points
        margin: Margin to add to bounding box
        N: Number of samples to generate
        method: Sampling method ('lhs' or 'halton')
        distance_threshold: Maximum distance to trajectory for points to keep
        rng_seed: Random seed for reproducibility
    
    Returns:
        tuple: (sampled_points, mask, bounds)
            - sampled_points: Array of shape (N, 3) containing all sampled points
            - mask: Boolean array indicating which points are within threshold
            - bounds: Array of shape (3, 2) containing bounds [[xmin,xmax], [ymin,ymax], [zmin,zmax]]
    """
    # Calculate bounding box
    bounds = np.array([
        [trajectory[:,i].min() - margin, trajectory[:,i].max() + margin]
        for i in range(3)
    ])
    
    # Initialize random number generator
    rng = np.random.default_rng(rng_seed)
    
    # Generate samples based on method
    if method.lower() == 'lhs':
        samples = lhs_sample(N, bounds, rng=rng)
    elif method.lower() == 'halton':
        samples = halton_sample(N, bounds, bases=(2, 3, 5), skip=64)
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    # Calculate distances to trajectory
    distances, _ = distances_to_curve(samples, trajectory)
    mask = distances <= distance_threshold
    
    return samples, mask, bounds

def plot_samples(trajectory: np.ndarray, 
                samples: np.ndarray,
                mask: np.ndarray,
                title: str = "3D Trajectory with Samples",
                distance_threshold: float = 0.2,
                elev: float = 24,
                azim: float = 35):
    """Plot trajectory and samples in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.scatter(trajectory[:,0], trajectory[:,1], trajectory[:,2], 
            linewidth=2, label="3D Trajectory")
    
    # Plot all samples
    ax.scatter(samples[:,0], samples[:,1], samples[:,2], 
              s=12, alpha=0.25, label="Samples (all)")
    
    # Plot samples within threshold
    ax.scatter(samples[mask,0], samples[mask,1], samples[mask,2],
              s=26, marker='^', label=f"Within band ({distance_threshold})")
    
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    ax.view_init(elev=elev, azim=azim)
    
    return fig, ax

# Example usage:
if __name__ == "__main__":
    # Generate example trajectory
    T = 800
    t = np.linspace(-1, 1, T)
    traj3d = np.stack([
        t,                    # x = t
        1 - t**2,            # y = 1 - t^2
        0.5 * np.sin(np.pi * t)  # z = 0.5*sin(πt)
    ], axis=1)  # Shape (M, N)
    
    # Sample using LHS
    samples_lhs, mask_lhs, bounds = sample_around_trajectory(
        traj3d,
        margin=0.2,
        N=1000,
        method='lhs', # 'halton' or 'lhs',
        distance_threshold=0.2,
        rng_seed=7
    )
    
    # Sample using Halton
    # samples_hal, mask_hal, bounds = sample_around_trajectory(
    #     traj3d,
    #     margin=0.2,
    #     N=800,
    #     method='halton',
    #     distance_threshold=0.2
    # )
    
    # Plot results
    plot_samples(traj3d, samples_lhs, mask_lhs, 
                title="3D Arch with LHS Coverage", azim=35)
    # plot_samples(traj3d, samples_hal, mask_hal, 
    #             title="3D Arch with Halton Coverage", azim=-35)
    
    plt.show()
