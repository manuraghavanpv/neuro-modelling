import numpy as np
import matplotlib.pyplot as plt

# Kuramoto model simulation
def kuramoto_model(N, omega, K, timesteps, dt):
    theta = np.random.uniform(0, 2*np.pi, N)  # Initial phases
    history = np.zeros((timesteps, N))  # History of phases

    for t in range(timesteps):
        # Update phases
        theta += omega + (K/N) * np.sum(np.sin(theta[:, np.newaxis] - theta[np.newaxis, :]), axis=1) * dt
        # Normalize phases to [0, 2*pi)
        theta = np.mod(theta, 2*np.pi)
        # Store history
        history[t] = theta

    return history

# Parameters
N = 100         # Number of oscillators
omega = 1.0     # Natural frequency
K = 1.5         # Coupling strength
timesteps = 1000  # Number of simulation steps
dt = 0.1        # Time step

# Run simulation
history = kuramoto_model(N, omega, K, timesteps, dt)

# Plotting
plt.figure(figsize=(10, 5))
for i in range(N):
    plt.plot(history[:, i], alpha=0.5)
plt.title('Neural Oscillator Simulation (Kuramoto Model)')
plt.xlabel('Time Steps')
plt.ylabel('Phase')
plt.show()

