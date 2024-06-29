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
