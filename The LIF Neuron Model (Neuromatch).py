#LIF NEURON MODEL: PART 1

#Imports
import numpy as np
import matplotlib.pyplot as plt

#Defining Parameters
t_max = 150e-3   # second
dt = 1e-3        # second
tau = 20e-3      # second
el = -60e-3      # milivolt
vr = -70e-3      # milivolt
vth = -50e-3     # milivolt
r = 100e6        # ohm
i_mean = 25e-11  # ampere

for step in range(10):
  t = step*dt
  i = i_mean * (1 + np.sin((2 * t * np.pi)/0.01))

#Printing pretty numbers
for step in range(10):
  t = step*dt
  i = i_mean*(1 + np.sin((2*t*np.pi)/0.01))
  print(f"The synaptic input current is {i:.4e} at time {t:.3f}")

#The discrete time integration part (Euler Method)
v = el #initializing the voltage to be leak potential (given in question)
for step in range(10):
  t = step*dt #see what time it is
  i = i = i_mean * (1 + np.sin((2 * t * np.pi)/0.01)) #see what the "i" is at this time
  #tau and the rest of the parameters are defined already, formula we know
  v = v + (dt/tau) * (el - v + r*i)
  print(f"The voltage is {v:.4e} at time {t:.3f}")

#Plotting synaptic input current
step_end = 25 #we want 25 points

with plt.xkcd(): #gives a Comic Sans look to the plot
  plt.figure() #initializing the figure
  plt.title("Synaptic Input Current")
  plt.xlabel('Time(s)')
  plt.ylabel("I(A)")

  for step in range(step_end):
    t = step * dt
    i = i_mean * (1 + np.sin((2 * t * np.pi)/0.01))
    plt.plot(t, i, 'ko') #ko gives black dots on the plot just like ro gives red

plt.show()

#Plotting Membrane Potential
step_end = int(t_max/dt) #t_max is defined initially and dividing it by dt gives step_end

plt.figure() #initializing the figure
plt.title("Membrane Voltage with sinusoidal i(t)")
plt.xlabel('Time(s)')
plt.ylabel("V_m(V)")

for step in range (step_end):
  t = step*dt
  i = i_mean * (1 + np.sin((2 * t * np.pi)/0.01))
  v = v + (dt/tau) * (el - v + r*i)
  plt.plot(t,v,'k.') #using k. gives even smaller markers and we need that here

plt.show()

#Random Synaptic Input
np.random.seed(2020) #Initializing the random number generator gives reproducibility to random draws
step_end = int(t_max/dt)
v = el

plt.figure() #initializing the figure
plt.title("Membrane Voltage with random i(t)")
plt.xlabel('Time(s)')
plt.ylabel("V_m(V)")

for step in range (step_end):
  t = step*dt
  random_number = 2 * np.random.random() - 1 #random number between 0 and 1 shifted to be between -1 and 1
  i = i_mean * (1 + (0.1 * ((t_max/dt)**0.5) * random_number))
  v = v + (dt/tau) * (el - v + r*i)
  plt.plot(t, v, 'k.') #k. gives smaller points

plt.show()

#Storing simulations in lists
np.random.seed(2020)
step_end = int(t_max/dt)
n = 50 #given 50 trials
v_n = [el] * n #initialize the list v_n with 50 values of membrane leak potential

plt.figure()
plt.title('Multiple realizations of $V_m$')
plt.xlabel('time(s)')
plt.ylabel('V_m(V)')

for step in range (step_end): #looping for step-end number of steps
  t = step * dt #finding value of t
  for j in range (0, n): #looping for n number of simulations
    i = i_mean * (1 + 0.1 * (t_max/dt)**(0.5)*(2* np.random.random() - 1)) #i for the corresponding time step with the random number between -1 and 1
    v_n[j] = v_n[j] + (dt/tau) * (el - v_n[j] + r*i)
  plt.plot([t]*n, v_n, 'k.', alpha = 0.1) #alpha helps control the opacity of the plot points and k. gives smaller points

plt.show()

#Plotting sample mean
np.random.seed(2020)
step_end = int(t_max / dt)
n = 50
v_n = [el] * n

plt.figure()
plt.title('Multiple realizations of $V_m$')
plt.xlabel('time (s)')
plt.ylabel('$V_m$ (V)')

for step in range(step_end):
  t = step * dt
  for j in range(0, n):
    i = i_mean * (1 + 0.1 * (t_max/dt)**(0.5) * (2* np.random.random() - 1))
    v_n[j] = v_n[j] + (dt / tau) * (el - v_n[j] + r*i)
  #Compute sample mean by summing list of v_n using sum, and dividing by n
  v_mean = sum(v_n)/n
  plt.plot(n*[t], v_n, 'k.', alpha=0.1)
  plt.plot(t, v_mean, 'C0.', alpha=0.8)
plt.show()

#Plotting sample standard deviation
np.random.seed(2020)
step_end = int(t_max / dt)
n = 50
v_n = [el] * n
with plt.xkcd():
  plt.figure()
  plt.title('Multiple realizations of $V_m$')
  plt.xlabel('time (s)')
  plt.ylabel('$V_m$ (V)')
  for step in range(step_end):
    t = step * dt
    for j in range(0, n):
      i = i_mean * (1 + 0.1 * (t_max/dt)**(0.5) * (2* np.random.random() - 1))
      v_n[j] = v_n[j] + (dt / tau) * (el - v_n[j] + r*i)
    v_mean = sum(v_n) / n

    # Initialize a list `v_var_n` with the contribution of each V_n(t) to
    v_var_n = [(v - v_mean)**2 for v in v_n] # Var(t) with a list comprehension over values of v_n
    v_var = sum(v_var_n) / (n - 1) # Compute sample variance v_var by summing the values of v_var_n with sum and dividing by n-1
    v_std = np.sqrt(v_var) # Compute the standard deviation v_std with the function np.sqrt
    plt.plot(n*[t], v_n, 'k.', alpha=0.1)
    plt.plot(t, v_mean, 'C0.', alpha=0.8, markersize=10) # Plot sample mean using alpha=0.8 and'C0.' for blue
    plt.plot(t, v_mean + v_std, 'C7.', alpha=0.8) # Plot mean + standard deviation with alpha=0.8 and argument 'C7.'
    plt.plot(t, v_mean - v_std, 'C7.', alpha=0.8) # Plot mean - standard deviation with alpha=0.8 and argument 'C7.'

  plt.show()

#Random synaptic input using numpy
np.random.seed(2020)
step_end = int(t_max / dt) - 1
t_range = np.linspace(0, t_max, num=step_end, endpoint=False)
v = el * np.ones(step_end)

i = i_mean * (1 + 0.1 * (t_max/dt) ** (0.5) * (2 * np.random.random(step_end) - 1)) #same as before
for step in range(1, step_end): # Loop for step_end steps
  v[step] = v[step-1] + (dt/tau)*(el - v[step-1] + r * i[step-1]) # Compute v as function of i

plt.figure()
plt.title('$V_m$ with random I(t)')
plt.xlabel('time (s)')
plt.ylabel('$V_m$ (V)')
plt.plot(t_range, v, 'k.')
plt.show()

#Using enumerate and indexing to get a continuous plot of Vm
np.random.seed(2020)
step_end = int(t_max / dt) - 1
t_range = np.linspace(0, t_max, num=step_end, endpoint=False)
v = el * np.ones(step_end)

i = i_mean * (1 + 0.1 * (t_max/dt) ** (0.5) * (2 * np.random.random(step_end) - 1))

for step, i_step in enumerate(i): # Loop for step_end values of i using enumerate
  if step==0: # Skip first iteration
    continue
  v[step] = v[step-1] + (dt/tau)*(el - v[step-1] + r * i_step) # Compute v as function of i using i_step

plt.figure()
plt.title('$V_m$ with random I(t)')
plt.xlabel('time (s)')
plt.ylabel('$V_m$ (V)')
plt.plot(t_range, v, 'k')
plt.show()

#Aggregation using 2D arrays
np.random.seed(2020)
step_end = int(t_max / dt)
n = 50
t_range = np.linspace(0, t_max, num=step_end)
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))

for step in range(1, step_end): # Loop for step_end - 1 steps
   v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step]) # Compute v_n

plt.figure()
plt.title('Multiple realizations of $V_m$')
plt.xlabel('time (s)')
plt.ylabel('$V_m$ (V)')
plt.plot(t_range, v_n.T, 'k', alpha=0.3)
plt.show()

#Aggregation: mean and standard deviation
np.random.seed(2020)
step_end = int(t_max / dt)
n = 50
t_range = np.linspace(0, t_max, num=step_end)
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))

for step in range(1, step_end):
  v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])

v_mean = np.mean(v_n, axis = 0) # Compute sample mean (use np.mean)
v_std = np.std(v_n, axis = 0) # Compute sample standard deviation (use np.std)

with plt.xkcd():
  plt.figure()
  plt.title('Multiple realizations of $V_m$')
  plt.xlabel('time (s)')
  plt.ylabel('$V_m$ (V)')
  plt.plot(t_range, v_n.T, 'k', alpha=0.3)
  plt.plot(t_range, v_n[-1], 'k', alpha=0.3, label='V(t)')
  plt.plot(t_range, v_mean, 'C0', alpha=0.8, label='mean')
  plt.plot(t_range, v_mean+v_std, 'C7', alpha=0.8)
  plt.plot(t_range, v_mean-v_std, 'C7', alpha=0.8, label='mean $\pm$ std')
  plt.legend()
  plt.show()

######################################################################################################################################


#LIF NEURON MODEL: PART 2

# Imports
import numpy as np
import matplotlib.pyplot as plt


#####################################################################

#1. Plotting a Histogram

np.random.seed(2020)

t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 10000
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))
nbins = 50

for step, t in enumerate(t_range):
  if step==0:
    continue
  v_n[:, step] =  v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])

plt.figure()
plt.ylabel('Frequency')
plt.xlabel('$V_m$ (V)')

plt.hist(v_n[:,int(step_end / 10)], nbins, histtype='stepfilled',linewidth=0, label=f't={t_max / 10} s') # Plot a histogram at t_max/10 (add labels and parameters histtype='stepfilled' and linewidth=0)
plt.hist(v_n[:, -1], nbins, histtype='stepfilled', linewidth=0, label=f't={t_max} s') # Plot a histogram at t_max (add labels and parameters histtype='stepfilled' and linewidth=0)

plt.legend()
plt.show()

#Spiking the LIF
# initialize the figure
plt.figure()

# collect axis of 1st figure in ax1
ax1 = plt.subplot(1, 2, 1)
plt.plot(t_range, my_data_left)
plt.ylabel('ylabel')

# share axis x with 1st figure
plt.subplot(1, 2, 2, sharey=ax1)
plt.plot(t_range, my_data_right)

# automatically adjust subplot parameters to figure
plt.tight_layout()
plt.show()

# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize spikes and spikes_n
spikes = {j: [] for j in range(n)}
spikes_n = np.zeros([step_end])

# Loop over time steps
for step, t in enumerate(t_range):

  # Skip first iteration
  if step == 0:
    continue

  # Compute v_n
  v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r*i[:, step])

  # Loop over simulations
  for j in range(n):

    # Check if voltage above threshold
    if v_n[j, step] >= vth:

      # Reset to reset voltage
      v_n[j, step] = vr

      # Add this spike time
      spikes[j] += [t]

      # Add spike count to this step
      spikes_n[step] += 1

# Collect mean Vm and mean spiking rate
v_mean = np.mean(v_n, axis=0)
spikes_mean =  spikes_n / n

# Initialize the figure
plt.figure()

# Plot simulations and sample mean
ax1 = plt.subplot(3, 1, 1)
for j in range(n):
  plt.scatter(t_range, v_n[j], color="k", marker=".", alpha=0.01)
plt.plot(t_range, v_mean, 'C1', alpha=0.8, linewidth=3)
plt.ylabel('$V_m$ (V)')

# Plot spikes
plt.subplot(3, 1, 2, sharex=ax1)
# for each neuron j: collect spike times and plot them at height j
for j in range(n):
  times = np.array(spikes[j])
  plt.scatter(times, j * np.ones_like(times), color="C0", marker=".", alpha=0.2)

plt.ylabel('neuron')

# Plot firing rate
plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(t_range, spikes_mean)
plt.xlabel('time (s)')
plt.ylabel('rate (Hz)')

plt.tight_layout()


#####################################################################


#2. Using Boolean Indexing
# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize spikes and spikes_n
spikes = {j: [] for j in range(n)}
spikes_n = np.zeros([step_end])

# Loop over time steps
for step, t in enumerate(t_range):

  # Skip first iteration
  if step == 0:
    continue

  # Compute v_n
  v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r*i[:, step])

  # Initialize boolean numpy array `spiked` with v_n > v_thr
  spiked = (v_n[:,step] >= vth)

  # Set relevant values of v_n to resting potential using spiked
  v_n[spiked,step] = vr

  # Collect spike times
  for j in np.where(spiked)[0]:
    spikes[j] += [t]
    spikes_n[step] += 1

# Collect mean spiking rate
spikes_mean = spikes_n / n

# Plot multiple realizations of Vm, spikes and mean spike rate
plot_all(t_range, v_n, spikes=spikes, spikes_mean=spikes_mean)



#####################################################################

#3. Boolean Indexing

# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize spikes and spikes_n
spikes = {j: [] for j in range(n)}
spikes_n = np.zeros([step_end])

# Loop over time steps
for step, t in enumerate(t_range):

  # Skip first iteration
  if step == 0:
    continue

  # Compute v_n
  v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r*i[:, step])

  # Initialize boolean numpy array `spiked` with v_n > v_thr
  spiked = (v_n[:,step] >= vth)

  # Set relevant values of v_n to resting potential using spiked
  v_n[spiked,step] = vr

  # Collect spike times
  for j in np.where(spiked)[0]:
    spikes[j] += [t]
    spikes_n[step] += 1

# Collect mean spiking rate
spikes_mean = spikes_n / n

# Plot multiple realizations of Vm, spikes and mean spike rate
plot_all(t_range, v_n, spikes=spikes, spikes_mean=spikes_mean)



#####################################################################

#4. Binary Raster Plot

# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize binary numpy array for raster plot
raster = np.zeros([n,step_end])

# Loop over time steps
for step, t in enumerate(t_range):

  # Skip first iteration
  if step==0:
    continue

  # Compute v_n
  v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r*i[:, step])

  # Initialize boolean numpy array `spiked` with v_n > v_thr
  spiked = (v_n[:,step] >= vth)

  # Set relevant values of v_n to v_reset using spiked
  v_n[spiked,step] = vr

  # Set relevant elements in raster to 1 using spiked
  raster[spiked,step] = 1

# Plot multiple realizations of Vm, spikes and mean spike rate
plot_all(t_range, v_n, raster)


#####################################################################

#5. Investigating Refractory Period
# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize binary numpy array for raster plot
raster = np.zeros([n,step_end])

# Initialize t_ref and last_spike
t_ref = 0.01
last_spike = -t_ref * np.ones([n])

# Loop over time steps
for step, t in enumerate(t_range):

  # Skip first iteration
  if step == 0:
    continue

  # Compute v_n
  v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r*i[:, step])

  # Initialize boolean numpy array `spiked` with v_n > v_thr
  spiked = (v_n[:,step] >= vth)

  # Set relevant values of v_n to v_reset using spiked
  v_n[spiked,step] = vr

  # Set relevant elements in raster to 1 using spiked
  raster[spiked,step] = 1.

  # Initialize boolean numpy array clamped using last_spike, t and t_ref
  clamped = (last_spike + t_ref > t)

  # Reset clamped neurons to vr using clamped
  v_n[clamped,step] = vr

  # Update numpy array last_spike with time t for spiking neurons
  last_spike[spiked] = t

# Plot multiple realizations of Vm, spikes and mean spike rate
plot_all(t_range, v_n, raster)


#####################################################################

#6. Functions
def ode_step(v, i, dt):
  """
  Evolves membrane potential by one step of discrete time integration

  Args:
    v (numpy array of floats)
      membrane potential at previous time step of shape (neurons)

    i (numpy array of floats)
      synaptic input at current time step of shape (neurons)

    dt (float)
      time step increment

  Returns:
    v (numpy array of floats)
      membrane potential at current time step of shape (neurons)
  """
  v = v + dt/tau * (el - v + r*i)

  return v


def spike_clamp(v, delta_spike):
  """
  Resets membrane potential of neurons if v>= vth
  and clamps to vr if interval of time since last spike < t_ref

  Args:
    v (numpy array of floats)
      membrane potential of shape (neurons)

    delta_spike (numpy array of floats)
      interval of time since last spike of shape (neurons)

  Returns:
    v (numpy array of floats)
      membrane potential of shape (neurons)
    spiked (numpy array of floats)
      boolean array of neurons that spiked  of shape (neurons)
  """

 
  # Boolean array spiked indexes neurons with v>=vth
  spiked = (v >= vth)
  v[spiked] = vr

  # Boolean array clamped indexes refractory neurons
  clamped = (t_ref > delta_spike)
  v[clamped] = vr

  return v, spiked


# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize binary numpy array for raster plot
raster = np.zeros([n,step_end])

# Initialize t_ref and last_spike
mu = 0.01
sigma = 0.007
t_ref = mu + sigma*np.random.normal(size=n)
t_ref[t_ref<0] = 0
last_spike = -t_ref * np.ones([n])

# Loop over time steps
for step, t in enumerate(t_range):

  # Skip first iteration
  if step==0:
    continue

  # Compute v_n
  v_n[:,step] = ode_step(v_n[:,step-1], i[:,step], dt)

  # Reset membrane potential and clamp
  v_n[:,step], spiked = spike_clamp(v_n[:,step], t - last_spike)

  # Update raster and last_spike
  raster[spiked,step] = 1.
  last_spike[spiked] = t

# Plot multiple realizations of Vm, spikes and mean spike rate
plot_all(t_range, v_n, raster)


#####################################################################
 #7. Making an LIF Class
 # Simulation class
class LIFNeurons:
  """
  Keeps track of membrane potential for multiple realizations of LIF neuron,
  and performs single step discrete time integration.
  """
  def __init__(self, n, t_ref_mu=0.01, t_ref_sigma=0.002,
               tau=20e-3, el=-60e-3, vr=-70e-3, vth=-50e-3, r=100e6):

    # Neuron count
    self.n = n

    # Neuron parameters
    self.tau = tau        # second
    self.el = el          # milivolt
    self.vr = vr          # milivolt
    self.vth = vth        # milivolt
    self.r = r            # ohm

    # Initializes refractory period distribution
    self.t_ref_mu = t_ref_mu
    self.t_ref_sigma = t_ref_sigma
    self.t_ref = self.t_ref_mu + self.t_ref_sigma * np.random.normal(size=self.n)
    self.t_ref[self.t_ref<0] = 0

    # State variables
    self.v = self.el * np.ones(self.n)
    self.spiked = self.v >= self.vth
    self.last_spike = -self.t_ref * np.ones([self.n])
    self.t = 0.
    self.steps = 0


  def ode_step(self, dt, i):

    # Update running time and steps
    self.t += dt
    self.steps += 1

    # One step of discrete time integration of dt
    self.v = self.v + dt / self.tau * (self.el - self.v + self.r * i)

    # Spike and clamp
    self.spiked = (self.v >= self.vth)
    self.v[self.spiked] = self.vr
    self.last_spike[self.spiked] = self.t
    clamped = (self.t_ref > self.t-self.last_spike)
    self.v[clamped] = self.vr

    self.last_spike[self.spiked] = self.t

# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize binary numpy array for raster plot
raster = np.zeros([n,step_end])

# Initialize neurons
neurons = LIFNeurons(n)

# Loop over time steps
for step, t in enumerate(t_range):

  # Call ode_step method
  neurons.ode_step(dt, i[:,step])

  # Log v_n and spike history
  v_n[:,step] = neurons.v
  raster[neurons.spiked, step] = 1.

# Report running time and steps
print(f'Ran for {neurons.t:.3}s in {neurons.steps} steps.')

# Plot multiple realizations of Vm, spikes and mean spike rate
plot_all(t_range, v_n, raster)
