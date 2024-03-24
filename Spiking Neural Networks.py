import numpy as np
import matplotlib.pyplot as plt

class SpikingNeuron:
    def __init__(self, tau_m=10, tau_ref=2, threshold=1, reset=0):
        self.tau_m = tau_m  # membrane time constant
        self.tau_ref = tau_ref  # refractory period
        self.threshold = threshold  # firing threshold
        self.reset = reset  # reset potential
        self.membrane_potential = 0  # current membrane potential
        self.refractory_time = 0  # current refractory time

    def update(self, dt, I):
        if self.refractory_time > 0:
            self.refractory_time -= dt  # decrease refractory time
            self.membrane_potential = self.reset  # reset potential during refractory period
        else:
            dVdt = (I - self.membrane_potential) / self.tau_m  # membrane potential change
            self.membrane_potential += dVdt * dt  # update membrane potential

            if self.membrane_potential >= self.threshold:
                self.membrane_potential = self.reset  # reset potential after firing
                self.refractory_time = self.tau_ref  # set refractory period

                return True  # neuron fired
        return False  # neuron did not fire

class SpikingNeuralNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.neurons = [SpikingNeuron() for _ in range(num_neurons)]
        self.spike_times = [[] for _ in range(num_neurons)]
        self.weights = np.random.uniform(low=0.05, high=0.1, size=(num_neurons, num_neurons))

    def simulate(self, dt, I_in):
        for t in range(len(I_in)):
            for i, neuron in enumerate(self.neurons):
                if neuron.update(dt, I_in[t]):
                    self.spike_times[i].append(t * dt)

            # STDP update
            for i in range(self.num_neurons):
                for j in range(self.num_neurons):
                    if i != j:
                        for spike_time in self.spike_times[i]:
                            if t * dt - spike_time <= 20:
                                self.weights[i][j] += 0.001  # increase weight if pre-synaptic neuron spikes before post-synaptic
                            else:
                                self.weights[i][j] -= 0.0005  # decrease weight if post-synaptic neuron spikes before pre-synaptic
                                self.weights[i][j] = max(0, self.weights[i][j])  # ensure weight is non-negative

# Simulation parameters
dt = 0.1
simulation_time = 100
num_neurons = 3

# Create spiking neural network
snn = SpikingNeuralNetwork(num_neurons)

# Input current (random for demonstration)
I_in = np.random.uniform(low=0, high=1, size=(int(simulation_time / dt)))

# Simulate network
snn.simulate(dt, I_in)

# Plotting
plt.figure(figsize=(10, 5))
for i in range(num_neurons):
    plt.plot(snn.spike_times[i], [i] * len(snn.spike_times[i]), '|')
plt.xlabel('Time')
plt.ylabel('Neuron index')
plt.title('Spiking Neural Network Activity')
plt.show()