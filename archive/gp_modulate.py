import numpy as np
from scipy.spatial.transform import Rotation
from typing import Callable, Optional, Tuple, List
from abc import ABC, abstractmethod
import sys
import os

# Add the path to import lpvds_class
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mani_py', 'src', 'se3_lpvds', 'src', 'lpvds', 'src'))
from src.se3_lpvds.src.lpvds.src.lpvds_class import lpvds_class


class GaussianProcessRegression:
    """Gaussian Process Regression implementation."""
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize GPR with specified input and output dimensions.
        
        Args:
            input_dim: Dimension of input data
            output_dim: Dimension of output data
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Training data storage
        self.input_data = np.empty((input_dim, 0))
        self.output_data = np.empty((output_dim, 0))
        self.n_data = 0
        
        # Hyperparameters
        self.l_scale = 3.2  # Length scale
        self.sigma_f = 1.0  # Signal variance
        self.sigma_n = 0.02  # Noise variance
        
        # Cache for regression
        self.KXX = None
        self.KXX_ = None
        self.alpha = None
        self.need_prepare = True
        
    def set_hyperparams(self, l: float, f: float, n: float):
        """Set hyperparameters for the GP."""
        self.l_scale = l
        self.sigma_f = f
        self.sigma_n = n
        
    def get_hyperparams(self) -> Tuple[float, float, float]:
        """Get current hyperparameters."""
        return self.l_scale, self.sigma_f, self.sigma_n
    
    def add_training_data(self, new_input: np.ndarray, new_output: np.ndarray):
        """
        Add a single training data point.
        
        Args:
            new_input: Input vector of shape (input_dim,)
            new_output: Output vector of shape (output_dim,)
        """
        new_input = new_input.reshape(-1, 1)
        new_output = new_output.reshape(-1, 1)
        
        if self.n_data == 0:
            self.input_data = new_input
            self.output_data = new_output
        else:
            self.input_data = np.hstack([self.input_data, new_input])
            self.output_data = np.hstack([self.output_data, new_output])
        
        self.n_data += 1
        self.need_prepare = True
        
    def add_training_data_batch(self, new_inputs: np.ndarray, new_outputs: np.ndarray):
        """
        Add multiple training data points at once.
        
        Args:
            new_inputs: Input matrix of shape (input_dim, n_samples)
            new_outputs: Output matrix of shape (output_dim, n_samples)
        """
        assert new_inputs.shape[1] == new_outputs.shape[1], "Number of samples must match"
        
        if self.n_data == 0:
            self.input_data = new_inputs
            self.output_data = new_outputs
        else:
            assert self.input_data.shape[0] == new_inputs.shape[0], "Input dimension mismatch"
            assert self.output_data.shape[0] == new_outputs.shape[0], "Output dimension mismatch"
            self.input_data = np.hstack([self.input_data, new_inputs])
            self.output_data = np.hstack([self.output_data, new_outputs])
        
        self.n_data = self.input_data.shape[1]
        self.need_prepare = True
    
    def sqe_cov_func_scalar(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Squared exponential covariance function for two vectors."""
        dist = x1 - x2
        d = np.dot(dist, dist)
        return self.sigma_f**2 * np.exp(-d / (2 * self.l_scale**2))
    
    def sqe_cov_func_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute covariance matrix for all pairs in X."""
        n_cols = X.shape[1]
        K = np.zeros((n_cols, n_cols))
        
        for i in range(n_cols):
            for j in range(i, n_cols):
                K[i, j] = self.sqe_cov_func_scalar(X[:, i], X[:, j])
                K[j, i] = K[i, j]
        
        return K
    
    def sqe_cov_func_vector(self, X: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute covariance vector between X and x."""
        n_cols = X.shape[1]
        k = np.zeros(n_cols)
        
        for i in range(n_cols):
            k[i] = self.sqe_cov_func_scalar(X[:, i], x)
        
        return k
    
    def prepare_regression(self, force_prepare: bool = False):
        """Prepare regression by computing necessary matrices."""
        if not self.need_prepare and not force_prepare:
            return
        
        if self.n_data == 0:
            return
        
        # Compute covariance matrix
        self.KXX = self.sqe_cov_func_matrix(self.input_data)
        self.KXX_ = self.KXX.copy()
        
        # Add measurement noise
        self.KXX_ += self.sigma_n**2 * np.eye(self.KXX_.shape[0])
        
        # Compute alpha for efficient prediction
        self.alpha = np.zeros_like(self.output_data)
        
        # Use Cholesky decomposition for efficiency
        L = np.linalg.cholesky(self.KXX_)
        
        for i in range(self.output_data.shape[0]):
            y = self.output_data[i, :]
            # Solve L * L^T * alpha = y
            alpha_i = np.linalg.solve(L.T, np.linalg.solve(L, y))
            self.alpha[i, :] = alpha_i
        
        self.need_prepare = False
    
    def do_regression(self, inp: np.ndarray, prepare: bool = False) -> np.ndarray:
        """
        Perform regression at a new input point.
        
        Args:
            inp: Input vector of shape (input_dim,)
            prepare: Whether to force preparation
            
        Returns:
            Output prediction of shape (output_dim,)
        """
        outp = np.zeros(self.output_dim)
        
        if self.n_data == 0:
            return outp
        
        self.prepare_regression(prepare)
        
        # Compute covariance with training data
        k_Xx = self.sqe_cov_func_vector(self.input_data, inp)
        
        # Compute prediction
        for i in range(self.output_dim):
            outp[i] = np.dot(k_Xx, self.alpha[i, :])
        
        return outp
    
    def clear_training_data(self):
        """Clear all training data."""
        self.input_data = np.empty((self.input_dim, 0))
        self.output_data = np.empty((self.output_dim, 0))
        self.n_data = 0
        self.need_prepare = True
        self.KXX = None
        self.KXX_ = None
        self.alpha = None


class LocallyModulatedDS(ABC):
    """Base class for Locally Modulated Dynamical Systems."""
    
    def __init__(self, original_dynamics: Optional[Callable] = None):
        """
        Initialize with original dynamics function.
        
        Args:
            original_dynamics: Function that maps position to velocity
        """
        self.original_dynamics = original_dynamics
    
    def set_original_dynamics(self, original_dynamics: Callable):
        """Set the original dynamics function."""
        self.original_dynamics = original_dynamics
    
    def get_original_dynamics(self) -> Callable:
        """Get the original dynamics function."""
        return self.original_dynamics
    
    @abstractmethod
    def modulation_function(self, position: np.ndarray) -> np.ndarray:
        """
        Compute the modulation matrix at a given position.
        Must be implemented by subclasses.
        """
        pass
    
    def get_output(self, position: np.ndarray) -> np.ndarray:
        """
        Get the modulated velocity at a given position.
        
        Args:
            position: Current position
            
        Returns:
            Modulated velocity
        """
        modulation = self.modulation_function(position)
        original_vel = self.original_dynamics(position)
        return modulation @ original_vel


class GaussianProcessModulatedDS(LocallyModulatedDS):
    """Implementation of GP-MDS (Gaussian Process Modulated Dynamical System)."""
    
    MIN_ANGLE = 0.001
    
    def __init__(self, original_dynamics: Callable):
        """
        Initialize GP-MDS with original dynamics.
        
        Args:
            original_dynamics: Function that maps position to velocity
        """
        super().__init__(original_dynamics)
        self.gpr = GaussianProcessRegression(3, 4)
        self.gpr.set_hyperparams(3.2, 1.0, 0.02)
    
    def compute_reshaping_parameters(self, actual_vel: np.ndarray, 
                                    original_vel: np.ndarray) -> np.ndarray:
        """
        Compute the reshaping parameters from actual and original velocities.
        
        Args:
            actual_vel: Actual/desired velocity
            original_vel: Original dynamics velocity
            
        Returns:
            4D vector containing axis-angle representation and speed scaling
        """
        theta = np.zeros(4)
        
        # Speed scaling with bias
        if np.linalg.norm(original_vel) > 1e-6:
            kappa = np.linalg.norm(actual_vel) / np.linalg.norm(original_vel) - 1
        else:
            kappa = 0
        
        # Compute rotation from original to actual velocity
        if np.linalg.norm(actual_vel) > 1e-6 and np.linalg.norm(original_vel) > 1e-6:
            # Normalize vectors
            v1 = original_vel / np.linalg.norm(original_vel)
            v2 = actual_vel / np.linalg.norm(actual_vel)
            
            # Compute rotation axis and angle
            axis = np.cross(v1, v2)
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            
            if np.linalg.norm(axis) > 1e-6:
                axis = axis / np.linalg.norm(axis)
                theta[:3] = axis * angle
        
        theta[3] = kappa
        return theta
    
    def modulation_function_from_params(self, angle_axis: np.ndarray, 
                                       speed_scaling: float) -> np.ndarray:
        """
        Compute modulation matrix from angle-axis and speed scaling.
        
        Args:
            angle_axis: 3D angle-axis representation
            speed_scaling: Speed scaling factor
            
        Returns:
            3x3 modulation matrix
        """
        angle = np.linalg.norm(angle_axis)
        
        if angle < self.MIN_ANGLE:
            modulation_matrix = np.eye(3)
        else:
            axis = angle_axis / angle
            # Rodrigues' rotation formula
            K = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
            modulation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        
        # Apply speed scaling (ensure it doesn't stop motion)
        speed_scaling_final = speed_scaling + 1.0
        # Could add thresholding here if needed:
        # speed_scaling_final = max(speed_scaling_final, 0.5)
        
        modulation_matrix *= speed_scaling_final
        return modulation_matrix
    
    def modulation_function(self, position: np.ndarray) -> np.ndarray:
        """
        Compute modulation matrix at a given position using GP regression.
        
        Args:
            position: 3D position vector
            
        Returns:
            3x3 modulation matrix
        """
        # Get reshaping parameters from GP
        theta_hat = self.gpr.do_regression(position)
        
        # Extract angle-axis and speed scaling
        angle_axis = theta_hat[:3]
        speed_scaling = theta_hat[3]
        
        return self.modulation_function_from_params(angle_axis, speed_scaling)
    
    def add_data(self, new_pos: np.ndarray, new_vel: np.ndarray):
        """
        Add a single training point.
        
        Args:
            new_pos: Position vector
            new_vel: Velocity vector
        """
        original_vel = self.original_dynamics(new_pos)
        reshaping_params = self.compute_reshaping_parameters(new_vel, original_vel)
        self.gpr.add_training_data(new_pos, reshaping_params)
    
    def add_data_batch(self, positions: np.ndarray, velocities: np.ndarray):
        """
        Add multiple training points at once.
        
        Args:
            positions: Position matrix (3 x n_samples) or list of position vectors
            velocities: Velocity matrix (3 x n_samples) or list of velocity vectors
        """
        # Handle list input
        if isinstance(positions, list):
            positions = np.column_stack(positions)
        if isinstance(velocities, list):
            velocities = np.column_stack(velocities)
        
        n_samples = positions.shape[1]
        reshaping_params = np.zeros((4, n_samples))
        
        for i in range(n_samples):
            pos = positions[:, i]
            vel = velocities[:, i]
            original_vel = self.original_dynamics(pos)
            reshaping_params[:, i] = self.compute_reshaping_parameters(vel, original_vel)
        
        self.gpr.add_training_data_batch(positions, reshaping_params)
    
    def clear_data(self):
        """Clear all training data."""
        self.gpr.clear_training_data()
    
    def get_gpr(self) -> GaussianProcessRegression:
        """Get the GPR object."""
        return self.gpr


class LPVDSWrapper:
    """Wrapper for lpvds_class to provide the same interface as LinearVelocityField."""
    
    def __init__(self, x: np.ndarray, x_dot: np.ndarray, x_att: np.ndarray):
        """
        Initialize LPVDS wrapper.
        
        Args:
            x: Training positions (N x D)
            x_dot: Training velocities (N x D)
            x_att: Attractor/target position (D,)
        """
        self.lpvds = lpvds_class(x, x_dot, x_att)
        self.lpvds.begin()  # Initialize the learning process
        self.x_att = x_att
    
    def compute_velocity(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute velocity at given position using LPVDS predict method.
        
        Args:
            pos: Current position (D,)
            
        Returns:
            Velocity vector (D,)
        """
        # lpvds.predict expects input as (M, N) where M is number of points, N is dimension
        pos_reshaped = pos.reshape(1, -1)  # Shape: (1, D)
        vel_pred = self.lpvds.predict(pos_reshaped)  # Returns (D, 1)
        return vel_pred.flatten()  # Return as (D,)
    
    def __call__(self, pos: np.ndarray) -> np.ndarray:
        """Make the object callable."""
        return self.compute_velocity(pos)
    
    def set_target(self, target: np.ndarray):
        """Update the target position."""
        self.x_att = target
        self.lpvds.x_att = target


# Example usage
if __name__ == "__main__":
    # Create sample training data for LPVDS
    D = 3
    target = np.zeros(D)
    
    # Generate some sample trajectory data
    n_samples = 50
    x_data = np.random.randn(n_samples, D) * 2.0  # Random positions
    # Create velocities pointing towards target with some noise
    x_dot_data = np.zeros((n_samples, D))
    for i in range(n_samples):
        direction = target - x_data[i, :]
        x_dot_data[i, :] = direction * 0.5 + np.random.randn(D) * 0.1
    
    # Create LPVDS wrapper
    lpvds_ds = LPVDSWrapper(x_data, x_dot_data, target)
    
    # Create GP-MDS
    gp_mds = GaussianProcessModulatedDS(lpvds_ds)
    
    # Add some training data
    pos1 = np.array([1.0, 0.0, 0.0])
    vel1 = np.array([0.0, 0.2, 0.0])  # Desired velocity different from LPVDS
    gp_mds.add_data(pos1, vel1)
    
    # Query the modulated system
    test_pos = np.array([1.0, 0.0, 0.0])
    modulated_vel = gp_mds.get_output(test_pos)
    original_vel = lpvds_ds(test_pos)
    
    print(f"Original velocity (LPVDS): {original_vel}")
    print(f"Modulated velocity: {modulated_vel}")
    
    # Example with more training data
    # Generate additional trajectory for another LPVDS
    x_data2 = np.random.randn(30, D) * 1.5
    x_dot_data2 = np.zeros((30, D))
    for i in range(30):
        direction = target - x_data2[i, :]
        x_dot_data2[i, :] = direction * 0.3 + np.random.randn(D) * 0.05
    
    lpvds_ds2 = LPVDSWrapper(x_data2, x_dot_data2, target)
    
    # Create another GP-MDS with LPVDS dynamics
    gp_mds_lpvds = GaussianProcessModulatedDS(lpvds_ds2)
    
    # Batch add data
    positions = np.random.randn(3, 5)
    velocities = np.random.randn(3, 5) * 0.1
    gp_mds_lpvds.add_data_batch(positions, velocities)
    
    print(f"\nNumber of training points: {gp_mds_lpvds.get_gpr().n_data}")