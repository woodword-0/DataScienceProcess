import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
####################################################################################
####################################################################################
# Example function
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))
####################################################################################
####################################################################################
# B.S. Data
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
####################################################################################
####################################################################################
# Evaluate
Z = f(X, Y)
####################################################################################
####################################################################################
# Create 3d-Plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()


# Display the plot using Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig)

####################################################################################
####################################################################################
# Find Critical Points
# dx, dy = np.gradient(Z)
# dx
# ####################################################################################
# ####################################################################################
# # Hessian and Curvature
# dxx, dxy = np.gradient(dx)
# dyx, dyy = np.gradient(dy)
# curvature = (dxx * dyy - dxy * dyx) / (1 + dx ** 2 + dy ** 2) ** 1.5
# ####################################################################################
# ####################################################################################
# # Extrema
# maxima = np.zeros_like(Z)
# minima = np.zeros_like(Z)
# saddle = np.zeros_like(Z)

# max_idx = curvature < 0
# min_idx = curvature > 0
# saddle_idx = curvature == 0

# maxima[max_idx] = Z[max_idx]
# minima[min_idx] = Z[min_idx]
# saddle[saddle_idx] = Z[saddle_idx]

# x_max = X[max_idx]
# y_max = Y[max_idx]
# z_max = Z[max_idx]

# ####################################################################################
# ####################################################################################
# # Plot Extrema on Function
# fig = plt.figure()
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
# # ax.set_title('Surface plot')
# # ax.set_xlabel('x')
# # ax.set_ylabel('y')
# # ax.set_zlabel('z')
# # plt.show()

# surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
# surf = ax.plot_surface(X, Y, curvature, cmap='magma', alpha=0.5)
# # surf = ax.plot_surface(x_max, y_max, z_max, color='red')
# # surf = ax.plot_surface(X, Y, minima[:5], color='green', alpha=0.5)
# # surf = ax.plot_surface(X, Y, saddle, color='black', alpha=0.5)

# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.show()
####################################################################################
####################################################################################
# Extrema

####################################################################################
####################################################################################
# Extrema

####################################################################################
####################################################################################
# Extrema
