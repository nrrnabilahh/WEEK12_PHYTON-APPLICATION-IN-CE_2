#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Python program to implement Runge Kutta method
# A sample differential equation "dy / dx = (x - y)/2"
def dydx(x, y):
    return ((x - y)/2)

# Finds value of y for a given x using step size h
# and initial value y0 at x0.
def rungeKutta(x0, y0, x, h):
    # Count number of iterations using step size or
    # step height
    n = int((x - x0)/h)
    # Iterate for number of iterations
    y = y0
    for i in range(1, n + 1):
        # Apply Runge Kutta Formulas to find next value of y
        k1 = h * dydx(x0, y)
        k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2)
        k4 = h * dydx(x0 + h, y + k3)
        
        # Update next value of y
        y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        
        # Update next value of x
        x0 = x0 + h
    return y

# Driver method
x0 = 0
y = 1
x = 5
h = 0.2
print('The value of y at x is:', rungeKutta(x0, y, x, h))


# In[2]:


# Importing necessary libraries
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

# parameter values for original SEIR
V = 1154 # m3
R = 8.314 # R[J/(mol*K)]
T = 120 # °C

mass_in = 473220 # kg/hr
MW_in = 56.6
mass_out = 28742 # kg/hr
MW_out = 40.4

# calculation, time in minutes
Tk = T + 273.15 # Convert to Kelvin
mol_in = mass_in/MW_in/60 # kmol/min
mol_out = mass_out/MW_out/60 # kmol/min

# initial condition
P0 = 1830 # kPa


# differential equation using ideal gas
def dPdt(P, t):
    # the differential equations
    dpdt = (mol_in - mol_out)/(V/(R*Tk))
    
    return dpdt


# create the x axis for the integration
# time to response is 20 minutes
start = 0
end = 30
t = np.linspace(start, end, end)

Pinitial = np.linspace(P0, P0, end)

# integration of the differential equation
P = sc.integrate.odeint(dPdt, P0, t)


# Plotting the results
plt.figure()
plt.plot(t, P, 'r', label='Pressure Increases')
plt.plot(t, Pinitial, 'b--', label='Original Pressure', linewidth=2)
plt.title(f'System Pressure Profile for total Volume of {V} m^3')
plt.xlabel('Time (minutes)')
plt.ylabel('Pressure (kPa)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.legend()
plt.show()


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
plt.ion()
plt.rcParams['figure.figsize'] = 10, 8

P = 0      # birth rate
d = 0.0001  # natural death percent (per day)
B = 0.0095  # transmission percent  (per day)
G = 0.0001  # resurect percent (per day)
A = 0.0001  # destroy percent  (per day)

# solve the system dy/dt = f(y, t)
def f(y, t):
     Si = y[0]
     Zi = y[1]
     Ri = y[2]
     # the model equations (see Munz et al. 2009)
     f0 = P - B*Si*Zi - d*Si
     f1 = B*Si*Zi + G*Ri - A*Si*Zi
     f2 = d*Si + A*Si*Zi - G*Ri
     return [f0, f1, f2]

# initial conditions
S0 = 500.              # initial population
Z0 = 0                 # initial zombie population
R0 = 0                 # initial death population
y0 = [S0, Z0, R0]     # initial condition vector
t  = np.linspace(0, 5., 1000)         # time grid

# solve the DEs
soln = odeint(f, y0, t)
S = soln[:, 0]
Z = soln[:, 1]
R = soln[:, 2]

# plot results
plt.figure()
plt.plot(t, S, label='Living')
plt.plot(t, Z, label='Zombies')
plt.xlabel('Days from outbreak')
plt.ylabel('Population')
plt.title('Zombie Apocalypse - No Init. Dead Pop.; No New Births.')
plt.legend(loc=0)

# change the initial conditions
R0 = 0.01*S0   # 1% of initial pop is dead
y0 = [S0, Z0, R0]

# solve the DEs
soln = odeint(f, y0, t)
S = soln[:, 0]
Z = soln[:, 1]
R = soln[:, 2]

plt.figure()
plt.plot(t, S, label='Living')
plt.plot(t, Z, label='Zombies')
plt.xlabel('Days from outbreak')
plt.ylabel('Population')
plt.title('Zombie Apocalypse - 1% Init. Pop. is Dead; No New Births.')
plt.legend(loc=0)

# change the initial conditions
R0 = 0.01*S0   # 1% of initial pop is dead
P  = 10        # 10 new births daily
y0 = [S0, Z0, R0]

# solve the DEs
soln = odeint(f, y0, t)
S = soln[:, 0]
Z = soln[:, 1]
R = soln[:, 2]

plt.figure()
plt.plot(t, S, label='Living')
plt.plot(t, Z, label='Zombies')
plt.xlabel('Days from outbreak')
plt.ylabel('Population')
plt.title('Zombie Apocalypse - 1% Init. Pop. is Dead; 10 Daily Births')
plt.legend(loc=0)


# In[9]:


# Import necessary libraries
import numpy as np
import scipy as sc
# Define the coefficient matrix A
A = np.array([[1, 3, 5], [2, 5, 1], [2, 3, 8]])
print(A)
# Define the right-hand side vector b
b = np.array([[10], [8], [3]])
print(b)
# Calculate the inverse of A and multiply by b to find the solution vector
C = sc.linalg.inv(A).dot(b)
print(C)
# Alternatively, use the solve function to find the solution vector directly
D = sc.linalg.solve(A, b)
print(D)


# In[10]:


import numpy as np
from scipy.linalg import solve
# Coefficients matrix
A = np.array([[3, 2],
              [1, 2]])
# Constants vector
b = np.array([1, 0])
# Using the solve function to find the solution to the system of equations
solution = solve(A, b)
solution


# In[12]:


A = np.array([[1, 3, 5], [2, 5, 1], [2, 3, 8]])
print(A)
# Define the right-hand side vector b
b = np.array([[10], [8], [3]])
print(b)
# Calculate the inverse of A and multiply by b to find the solution vector
C = sc.linalg.inv(A).dot(b)
print(C)
# Alternatively, use the solve function to find the solution vector directly
D = sc.linalg.solve(A, b)
print(D)


# In[18]:


b = np.array([[10], [8], [3]])
print(b)


# In[ ]:




