import numpy as np
import matplotlib.pyplot as plt

class IntegrateAndFireNeuron:
    def __init__(self, threshold=1.0, reset_voltage=0.0, membrane_resistance=1.0, membrane_time_constant=10.0):
        self.threshold = threshold  # Threshold voltage for firing
        self.reset_voltage = reset_voltage  # Voltage reset after firing
        self.membrane_resistance = membrane_resistance  # Membrane resistance
        self.membrane_time_constant = membrane_time_constant  # Membrane time constant
        self.membrane_voltage = reset_voltage  # Initialize membrane voltage
        self.spike_times = []  # List to store spike times

    def integrate(self, input_current, time_step):
        # Compute change in membrane voltage using the membrane equation
        delta_voltage = (input_current / self.membrane_resistance - self.membrane_voltage) / self.membrane_time_constant * time_step
        self.membrane_voltage += delta_voltage

        # Checking if threshold is crossed
        if self.membrane_voltage >= self.threshold:
            self.membrane_voltage = self.reset_voltage  # Reset membrane voltage
            self.spike_times.append(time_step)  # Record spike time

    def simulate(self, input_currents, time_steps):
        for i in range(len(time_steps)):
            self.integrate(input_currents[i], time_steps[i])

    def plot_voltage(self, time_steps):
        plt.plot(time_steps, [self.reset_voltage] * len(time_steps), 'r--', label='Reset Voltage')
        plt.plot(time_steps, [self.threshold] * len(time_steps), 'g--', label='Threshold')
        plt.plot(time_steps, [self.membrane_voltage for _ in time_steps], label='Membrane Voltage')
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage')
        plt.title('Membrane Potential over Time')
        plt.legend()
        plt.show()

# Simulation Parameters
total_time = 100  # Total simulation time in milliseconds
time_step = 0.1  # Time step size in milliseconds
num_steps = int(total_time / time_step)  # Number of time steps

# Step Input (Input Current)
input_currents = np.zeros(num_steps)
input_currents[int(num_steps/4):int(num_steps*3/4)] = 0.5  # Injecting current for a portion of time

# Time steps
time_steps = np.linspace(0, total_time, num_steps)

# Neuron object
neuron = IntegrateAndFireNeuron(threshold=0.8, reset_voltage=0.0, membrane_resistance=1.0, membrane_time_constant=10.0)

# Neuron Simulation
neuron.simulate(input_currents, time_steps)

# Membrane potential over time
neuron.plot_voltage(time_steps)
