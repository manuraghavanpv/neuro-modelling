import numpy as np
import matplotlib.pyplot as plt

# Constants
C_m = 1.0    # membrane capacitance (uF/cm^2)
g_Na = 120.0 # maximum sodium conductance (mS/cm^2)
g_K = 36.0   # maximum potassium conductance (mS/cm^2)
g_L = 0.3    # leak conductance (mS/cm^2)
E_Na = 50.0  # sodium reversal potential (mV)
E_K = -77.0  # potassium reversal potential (mV)
E_L = -54.387# leak reversal potential (mV)

# Helper functions for gating variables
def alpha_m(V):
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

def alpha_n(V):
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)

# Hodgkin-Huxley model
def HH_model(I, V_init, dt, T):
    # Number of time steps
    num_steps = int(T / dt)

    # Initialize arrays
    V = np.zeros(num_steps)
    m = np.zeros(num_steps)
    h = np.zeros(num_steps)
    n = np.zeros(num_steps)
    time = np.linspace(0, T, num_steps)

    # Initial conditions
    V[0] = V_init
    m[0] = alpha_m(V_init) / (alpha_m(V_init) + beta_m(V_init))
    h[0] = alpha_h(V_init) / (alpha_h(V_init) + beta_h(V_init))
    n[0] = alpha_n(V_init) / (alpha_n(V_init) + beta_n(V_init))

    # Simulation loop
    for i in range(1, num_steps):
        # Update gating variables
        m[i] = m[i - 1] + dt * (alpha_m(V[i - 1]) * (1 - m[i - 1]) - beta_m(V[i - 1]) * m[i - 1])
        h[i] = h[i - 1] + dt * (alpha_h(V[i - 1]) * (1 - h[i - 1]) - beta_h(V[i - 1]) * h[i - 1])
        n[i] = n[i - 1] + dt * (alpha_n(V[i - 1]) * (1 - n[i - 1]) - beta_n(V[i - 1]) * n[i - 1])

        # Calculate membrane potential
        g_Na_m = g_Na * m[i] ** 3 * h[i]
        g_K_n = g_K * n[i] ** 4
        g_L_l = g_L

        I_Na = g_Na_m * (V[i - 1] - E_Na)
        I_K = g_K_n * (V[i - 1] - E_K)
        I_L = g_L_l * (V[i - 1] - E_L)

        # Total membrane current
        I_total = I_Na + I_K + I_L + I

        # Update membrane potential
        V[i] = V[i - 1] + dt * (1.0 / C_m) * (-I_total)

    return time, V

# Simulation parameters
I = 10.0      # input current (uA/cm^2)
V_init = -65  # initial membrane potential (mV)
dt = 0.01     # time step (ms)
T = 50        # total simulation time (ms)

# Run simulation
time, V = HH_model(I, V_init, dt, T)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(time, V, label='Membrane Potential (mV)')
plt.title('Hodgkin-Huxley Neuron Model')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.legend()
plt.grid(True)
plt.show()