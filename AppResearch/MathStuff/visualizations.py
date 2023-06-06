import random
import numpy as np
from sympy import symbols, Eq, solve

def generate_random_integers():
    # Generate four random integers in the range [-10, 10]
    integers = [random.randint(-10, 10) for _ in range(4)]
    return integers

def create_matrix(integers):
    # Create a 2x2 matrix using the generated integers
    matrix = np.reshape(integers, (2, 2))
    return matrix

def check_determinant(matrix):
    # Check if the determinant of the matrix is equal to 1
    determinant = np.linalg.det(matrix)
    return determinant == 1

def find_equation(integers):
    # Find an equation that satisfies the conditions
    a, b, c, d = symbols('a b c d')
    equation = Eq(a*d - b*c, 1)  # Equation based on determinant
    solution = solve(equation.subs({a: integers[0], b: integers[1], c: integers[2], d: integers[3]}))
    return solution

# Generate random integers and create matrix until determinant is 1
while True:
    integers = generate_random_integers()
    matrix = create_matrix(integers)
    if check_determinant(matrix):
        equation = find_equation(integers)
        break

print("Generated Integers:", integers)
print("Matrix:")
print(matrix)
print("Equation:", equation)



import numpy as np
import matplotlib.pyplot as plt

# Fix the value of d
d = 2

# Define the range of integer values for b and c
b_values = np.arange(-10, 11)
c_values = np.arange(-10, 11)

# Create a grid of values for b and c
b_grid, c_grid = np.meshgrid(b_values, c_values)

# Compute the corresponding values of a using the equation
a_grid = (1 + b_grid * c_grid) / d

# Plot the graph
plt.figure(figsize=(8, 6))
plt.imshow(a_grid, cmap='RdYlBu', extent=[-10, 10, -10, 10], origin='lower')
plt.colorbar(label='a')
plt.xlabel('b')
plt.ylabel('c')
plt.title('Graph of a = (1 + b * c) / d (Fixed d = 2)')
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fix the value of d
d = 2

# Define the range of integer values for b and c
b_values = np.arange(-10, 11)
c_values = np.arange(-10, 11)

# Create a grid of values for b and c
b_grid, c_grid = np.meshgrid(b_values, c_values)

# Compute the corresponding values of a using the equation
a_grid = (-1 + b_grid * c_grid) / d

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(b_grid, c_grid, a_grid, cmap='RdYlBu')
ax.set_xlabel('b')
ax.set_ylabel('c')
ax.set_zlabel('a')
ax.set_title('Surface Plot of a = (1 + b * c) / d')

# Adjust viewing angle
ax.view_init(elev=30, azim=-45)

# Show the plot
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the range of integer values for b and c
b_values = np.arange(-10, 11)
c_values = np.arange(-10, 11)

# Create a grid of values for b and c
b_grid, c_grid = np.meshgrid(b_values, c_values)

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Initialize the plot
plot = ax.plot_surface(b_grid, c_grid, np.zeros_like(b_grid), cmap='RdYlBu')

# Animation update function
def update(frame):
    d = frame + 1  # Increment d for each frame
    a_grid = (1 + b_grid * c_grid) / d
    plot.set_array(a_grid.ravel())

# Create the animation
animation = FuncAnimation(fig, update, frames=10, interval=500, repeat=True)
animation
# Set the axis labels and title
ax.set_xlabel('b')
ax.set_ylabel('c')
ax.set_zlabel('a')
ax.set_title('3D Animation of a = (1 + b * c) / d')

# Adjust the viewing angle
ax.view_init(elev=30, azim=-45)

# Show the animation
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the range of integer values for b and c
b_values = np.arange(-10, 11)
c_values = np.arange(-10, 11)

# Create a grid of values for b and c
b_grid, c_grid = np.meshgrid(b_values, c_values)

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Initialize the plot
plot = ax.plot_surface(b_grid, c_grid, np.zeros_like(b_grid), cmap='RdYlBu')

# Animation update function
def update(frame):
    d = frame + 1  # Increment d for each frame
    a_grid = (1 + b_grid * c_grid) / d
    plot.set_array(a_grid.ravel())

# Create the animation
animation = FuncAnimation(fig, update, frames=10, interval=500, repeat=True)
plt.show()  # Include plt.show() here









import matplotlib.pyplot as plt
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D

# Define the list of ordered items
ordered_items = [
    "item1", "item2", "item3", "item4", "item5",
    "item6", "item7", "item8", "item9", "item10"
]

# Generate a word cloud from the ordered items
wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=10).generate(" ".join(ordered_items))
wordcloud
# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the word cloud as a set of 3D points
for i, (word, freq) in enumerate(wordcloud.words_.items()):
    x = i % 5
    y = i // 5
    z = freq
    ax.text(x, y, z, word, fontsize=15, ha='center', va='center')

# Set axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Frequency')
ax.set_title('3D Recommender Word Cloud')

# Adjust viewing angle
ax.view_init(elev=30, azim=-45)

# Hide axis ticks and labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Show the plot
plt.show()





