import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate a Point Cloud of a Torus
R = 3  # Major radius
r = 1  # Minor radius
theta = np.linspace(0, 2 * np.pi, 100)  # Angle parameter
phi = np.linspace(0, 2 * np.pi, 100)  # Angle parameter
theta, phi = np.meshgrid(theta, phi)
x = (R + r * np.cos(theta)) * np.cos(phi)
y = (R + r * np.cos(theta)) * np.sin(phi)
z = r * np.sin(theta)

# Step 2: Visualize the Torus
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


import numpy as np
from gudhi import persistence_graphical_tools as pg
from gudhi.representations import PersistenceIntervals
import matplotlib.pyplot as plt

# Step 1: Generate a Point Cloud of a Torus
R = 3  # Major radius
r = 1  # Minor radius
theta = np.linspace(0, 2 * np.pi, 100)  # Angle parameter
phi = np.linspace(0, 2 * np.pi, 100)  # Angle parameter
theta, phi = np.meshgrid(theta, phi)
x = (R + r * np.cos(theta)) * np.cos(phi)
y = (R + r * np.cos(theta)) * np.sin(phi)
z = r * np.sin(theta)

# Step 2: Compute Persistence Intervals
persistence = PersistenceIntervals()
persistence.fit([(x.flatten(), y.flatten(), z.flatten())])

# Step 3: Compute and Plot Persistence Landscapes
landscapes = pg.persistence_landscapes(persistence, homology_dimensions=[0, 1], num_landscapes=2)

# Display Persistence Landscapes
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
pg.plot_landscape(landscapes[0], ax=axs[0], title='Homology Class 1')
pg.plot_landscape(landscapes[1], ax=axs[1], title='Homology Class 2')
plt.show()
