from matplotlib.pylab import seed
import numpy as np
from numpy.random import seed
from scipy.integrate import odeint
from typing import Tuple, Dict

class PendulumGenerator:
    def __init__(self,g:float=9.81,l: float=1.0,dt :float=0.01,t_max: float=10.0):
        self.g = g
        self.l = l
        self.dt = dt
        self.t_max = t_max

    def single_pendulum(self, t: np.ndarray, y0: Tuple[float, float], damping: float=0.0) -> np.ndarray:
        """
        Generate trajectory data for a single pendulum.

        Parameters:
        - t: Time array.
        - y0: Initial state [theta, theta_dot].
        - damping: Damping coefficient.

        Returns:
        - data: Array of shape (len(t), 2) containing [theta, theta_dot] at each time step.
        """
       
        def dynamics(y,t):
            theta, theta_dot = y
            theta_ddot = -(self.g / self.l) * np.sin(theta) - damping * theta_dot
            return [theta_dot, theta_ddot]  
         
        return odeint(dynamics, y0, t) #odenit is scipy numerical integrator..odenit steps through every single point in the time array t ,uses dynamics function to calculate how much the pendulum should move and return a 2D array 
    

    def double_pendulum(
        self,
        t: np.ndarray,
        y0: Tuple[float, float, float, float],
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
    ) -> np.ndarray:
        """
        Generate trajectory data for a double pendulum.

        Parameters:
        - t: Time array.
        - y0: Initial state [theta1, theta1_dot, theta2, theta2_dot].
        - m1: Mass of the first pendulum.
        - m2: Mass of the second pendulum.
        - l1: Length of the first pendulum.
        - l2: Length of the second pendulum.

        Returns:
        - data: Array of shape (len(t), 4) containing [theta1, theta1_dot, theta2, theta2_dot] at each time step.
        """
        
        def dynamics(y,t):
            theta1, theta1_dot, theta2, theta2_dot = y

            denom=2*m1+m2-m2*np.cos(2*theta1-2*theta2)

            theta1_ddot = (
                -self.g * (2*m1 + m2) * np.sin(theta1)
                - m2 * self.g * np.sin(theta1 - 2*theta2)
                - 2 * np.sin(theta1 - theta2) * m2 * (
                    theta2_dot**2 * l2 + theta1_dot**2 * l1 * np.cos(theta1 - theta2)
                )
            ) / (l1 * denom)
            
            theta2_ddot = (
                2 * np.sin(theta1 - theta2) * (
                    theta1_dot**2 * l1 * (m1 + m2)
                    + self.g * (m1 + m2) * np.cos(theta1)
                    + theta2_dot**2 * l2 * m2 * np.cos(theta1 - theta2)
                )
            ) / (l2 * denom)

            return [theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]

        return odeint(dynamics, y0, t)


    def add_noise(self, data: np.ndarray, noise_level: float=0.01,seed:int=None) -> np.ndarray:
        """
        Add Gaussian noise to the generated data.

        Parameters:
        - data: Original data array.
        - noise_level: Standard deviation of the Gaussian noise.

        Returns:
        - noisy_data: Data array with added noise.
        """

        if seed is not None:
            np.random.seed(seed)

        if noise_level ==0:
            return data.copy()    

        signal_range = np.max(np.abs(data), axis=0)
        noise_std = noise_level * signal_range
        noise = np.random.normal(0, noise_std, data.shape)  #creates a matrix of random numbers with the same shape as data, with mean 0 and std noise_std
        
        return data + noise


def generate_dataset(
    system: str = "single",
    noise_levels: list = None,
    n_trajectories: int = 1,  #How many separate simulation runs you want for each noise level.
    **kwargs,
) -> Dict[float, list]: # Changed type hint: keys are floats, values are lists of arrays
    
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.05, 0.1]
    
    gen = PendulumGenerator()
    t = np.linspace(0, 10, 500)
    
    dataset = {}
    for noise_level in noise_levels:
        trajectories = []
        for traj_id in range(n_trajectories):
            if system == "single":
                y0 = [np.pi/3 + 0.1*traj_id, 0.0]  #during every simulations..the initial angle of the pendulum is varied by 0.1*traj_id to create different trajectories
                # Pass kwargs here so damping can be configured
                data = gen.single_pendulum(t, y0, **kwargs)
            else:  
                # Added traj_id variance to the double pendulum angles
                y0 = [np.pi/4 + 0.1*traj_id, 0.0, np.pi/6 - 0.05*traj_id, 0.0]  #the first argument ..starting angle of the top arm ,2nd argument -ang velocity of top arm..and similar things for the rest.
                data = gen.double_pendulum(t, y0, **kwargs)
            
            data_noisy = gen.add_noise(data, noise_level, seed=42+traj_id)  #every trajectory has a different seed to ensure different noise patterns
            trajectories.append(data_noisy)
        
        # Store as a list of trajectories to prevent SINDy boundary errors
        dataset[noise_level] = trajectories
    
    return dataset



    #Loop 0 (traj_id = 0):y0 becomes [1.047, 0.0] ($60^\circ$). The physics engine runs, calculates the time-series data, and saves it.Loop 1 (traj_id = 1):y0 becomes [1.147, 0.0] ($65.7^\circ$). The engine runs again from this higher drop point and saves the new data.Loop 2 (traj_id = 2):y0 becomes [1.247, 0.0] ($71.4^\circ$). The engine runs a third time from an even higher drop point.All three runs are then collected, noise is added to each, and the SINDy model gets a diverse set of physics data to study.