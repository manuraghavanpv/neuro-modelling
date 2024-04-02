import numpy as np
import matplotlib.pyplot as plt

def drift_diffusion_model(drift_rate, boundary, noise, dt, num_trials):
    """Simulate drift-diffusion model for decision making."""
    num_steps = int(boundary / drift_rate * 10)  # Ensure enough steps to reach boundary
    steps = np.random.normal(drift_rate * dt, noise * np.sqrt(dt), (num_trials, num_steps))
    accumulated_evidence = np.cumsum(steps, axis=1)
    decisions = np.argmax(np.abs(accumulated_evidence) >= boundary, axis=1)
    rt = np.argmax(np.abs(accumulated_evidence) >= boundary, axis=1) * dt
    return decisions, rt

# Parameters
drift_rate = 0.1  # Drift rate
boundary = 1.0    # Decision boundary
noise = 0.1       # Noise
dt = 0.01         # Time step
num_trials = 1000 # Number of trials

# Drift-diffusion model simulation
decisions, rt = drift_diffusion_model(drift_rate, boundary, noise, dt, num_trials)

# Response times (histogram)
plt.hist(rt, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Response Time')
plt.ylabel('Frequency')
plt.title('Distribution of Response Times')
plt.show()

# Decisions (histogram)
plt.hist(decisions, bins=3, color='lightgreen', edgecolor='black', alpha=0.7)
plt.xlabel('Decision')
plt.ylabel('Frequency')
plt.title('Distribution of Decisions')
plt.xticks([0, 1], ['Option 1', 'Option 2'])
plt.show()
