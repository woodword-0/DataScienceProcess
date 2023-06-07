import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from matplotlib.animation import FuncAnimation

####################################################################################
####################################################################################
# Example function
def f(x, y, t):
    return np.sin(np.sqrt(x ** 2 + y ** 2 + t))

####################################################################################
####################################################################################
# B.S. Data
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
# Evaluate
# Z = f(X, Y,0)
####################################################################################
####################################################################################
# Create 3d-Plot
fig = plt.figure()
ax = plt.axes(projection='3d')
line = ax.plot_surface(X, Y, f(X, Y, 0), cmap='viridis', edgecolor='none')

ax.set_title('Surface plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

def update(t):
    line._offsets3d = (X.flatten(), Y.flatten(), f(X, Y, t).flatten())

anim = FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), interval=100)
anim.save('animationattempt1')
plt.show()
# Display the animation using Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(anim.to_jshtml())





from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Example function
def f(x, y, t):
    return np.sin(np.sqrt(x ** 2 + y ** 2 + t))

####################################################################################
####################################################################################
# B.S. Data
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Create the figure and animation
fig = plt.figure()
ax = plt.axes(projection='3d')
line = ax.plot_surface(X, Y, f(X, Y, 0), cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

def update(t):
    line._offsets3d = (X.flatten(), Y.flatten(), f(X, Y, t).flatten())

anim = FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), interval=100)
plt.show()







import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

# Set up the ball
ball, = ax.plot([], [], 'o', color='r', markersize=10)

# Define the trajectory function
def trajectory(t):
    x = t
    y = 0.5 * 9.81 * t**2 # Parabolic trajectory with acceleration 9.81 m/s^2
    return x, y

# Define the update function for the animation
def update(frame):
    x, y = trajectory(frame)
    ball.set_data(x, y)
    return ball,

# Set up the animation
animation = FuncAnimation(fig, update, frames=np.arange(0, 10, 0.1), interval=1000)

# Show the animation
plt.axis('equal')
plt.xlim(0, 5)
plt.ylim(0, 15)
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Vertical Distance (m)')
plt.title('Ball Trajectory')
plt.show()
