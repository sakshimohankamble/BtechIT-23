import csv
import numpy as np

def sphere_function(x):
    # Calculate the sum of squared components
    return np.sum(np.power(x, 2))

def initialize_population(population_size, dimension, search_space):
    # Initialize the population within the search space
    population = np.random.uniform(search_space[0], search_space[1], (population_size, dimension))
    return population

def evaluate_population(population):
    # Evaluate the fitness of each solution in the population
    fitness_values = np.apply_along_axis(sphere_function, 1, population)
    return fitness_values

def cheetah_optimization_algorithm(population_size, dimension, search_space, max_iterations):
    population = initialize_population(population_size, dimension, search_space)
    best_solution = None
    best_fitness = np.inf

    # Create and open the CSV file
    filename = "coa_results.csv"
    csvfile = open(filename, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(["Iteration", "Best Solution", "Best Fitness"])

    for iteration in range(max_iterations):
        fitness_values = evaluate_population(population)

        # Update the best solution
        min_fitness = np.min(fitness_values)
        min_index = np.argmin(fitness_values)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_solution = population[min_index]

        # Append current iteration results to the CSV file
        writer.writerow([iteration, *best_solution, best_fitness])

        # Generate new candidate solutions using search operators
        # You can use various operators like mutation, crossover, etc.

        # Update the population with the new solutions

    # Close the CSV file
    csvfile.close()

    return best_solution, best_fitness

# User input for required parameters
population_size = int(input("Enter the population size: "))
dimension = int(input("Enter the dimension: "))
search_space_low = float(input("Enter the lower bound of the search space: "))
search_space_high = float(input("Enter the upper bound of the search space: "))
max_iterations = int(input("Enter the maximum number of iterations: "))

search_space = (search_space_low, search_space_high)

# Run the Cheetah Optimization Algorithm
best_solution, best_fitness = cheetah_optimization_algorithm(population_size, dimension, search_space, max_iterations)

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
