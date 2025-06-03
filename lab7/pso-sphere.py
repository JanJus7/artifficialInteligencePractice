# Import modules
import numpy as np
from matplotlib import pyplot as plt
# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
x_min = np.zeros(6)
x_max = np.ones(6)
my_bounds = (x_min, x_max)

def endurance(x):
    return -(np.sum(x**2) + np.sin(np.sum(x)))

def f(X):
    n_particles = X.shape[0]
    results = np.zeros(n_particles)
    for i in range(n_particles):
        results[i] = endurance(X[i])
    return results



# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)

# Perform optimization
# cost, pos = optimizer.optimize(fx.sphere, iters=1000)
cost, pos = optimizer.optimize(f, iters=1000)

# Obtain cost history from optimizer instance
cost_history = optimizer.cost_history

# Plot!
plot_cost_history(cost_history)
plt.show()