import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Rastrigin function
def rastrigin(x):
    return 10 * len(x) + sum([(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])

# Simulated Annealing Algorithm with tracking history
def simulated_annealing(objective, bounds, n_iterations, initial_temp, alpha):
    # Initialize solution
    dim = len(bounds)
    current_solution = np.random.uniform(bounds[:, 0], bounds[:, 1], dim)
    current_energy = objective(current_solution)
    best_solution = current_solution
    best_energy = current_energy
    temp = initial_temp

    # Track history for visualization
    solutions_history = [current_solution.copy()]
    
    for i in range(n_iterations):
        # Generate a new solution by adding a small random noise
        candidate = current_solution + np.random.uniform(-1, 1, dim)
        candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])  # keep within bounds
        candidate_energy = objective(candidate)

        # Calculate the change in energy
        delta_energy = candidate_energy - current_energy

        # Acceptance criteria
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temp):
            current_solution = candidate
            current_energy = candidate_energy

            # Update the best solution found
            if current_energy < best_energy:
                best_solution = current_solution
                best_energy = current_energy

        # Cool the temperature
        temp *= alpha

        # Append the current solution to history
        solutions_history.append(current_solution.copy())

    return best_solution, best_energy, solutions_history

# Parameters
bounds = np.array([[-5.12, 5.12], [-5.12, 5.12]])  # bounds for Rastrigin function (2D)
n_iterations = 500
initial_temp = 100
alpha = 0.98

# Run simulated annealing
best_sol, best_val, history = simulated_annealing(rastrigin, bounds, n_iterations, initial_temp, alpha)

# Extract history for plotting
x_history = [sol[0] for sol in history]
y_history = [sol[1] for sol in history]
z_history = [rastrigin(sol) for sol in history]

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Create a grid for the Rastrigin function surface
X = np.linspace(bounds[0, 0], bounds[0, 1], 100)
Y = np.linspace(bounds[1, 0], bounds[1, 1], 100)
X, Y = np.meshgrid(X, Y)
Z = rastrigin([X, Y])

# Plot the surface
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6, edgecolor="none")

# Plot the path of solutions
ax.plot(x_history, y_history, z_history, color="red", marker="o", markersize=3, label="Solution Path")
ax.scatter(*best_sol, best_val, color="blue", marker="*", s=100, label="Best Solution")

# Labels and legend
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Objective Value")
ax.set_title("Simulated Annealing on Rastrigin Function")
ax.legend()
plt.show()

