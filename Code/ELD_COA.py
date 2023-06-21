import csv
import numpy as np
import random

# Cheetah Optimization Algorithm
def cheetah_optimization():
    # Initialize the population
    def initialize_population():
        population = []
        for _ in range(population_size):
            solution = []
            for i in range(num_generators):
                solution.append(random.uniform(p_min[i], p_max[i]))
            population.append(solution)
        return population

    # Evaluate the fitness of the population
    def evaluate_fitness(population):
        fitness_values = []
        for solution in population:
            cost = calculate_total_cost(solution)
            fitness_values.append(cost)
        return fitness_values

    # Calculate the total cost of a solution
    def calculate_total_cost(solution):
        total_cost = 0
        for i in range(num_generators):
            cost = a[i] * solution[i] ** 2 + b[i] * solution[i] + c[i]
            total_cost += cost
        return total_cost

    # Select a parent based on fitness values using roulette wheel selection
    def select_parent(population, fitness_values):
        total_fitness = sum(fitness_values)
        probabilities = [fitness / total_fitness for fitness in fitness_values]
        return population[np.random.choice(range(population_size), p=probabilities)]

    # Perform crossover between two parents
    def crossover(parent1, parent2):
        offspring = []
        for i in range(num_generators):
            if random.random() < 0.5:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
        return offspring

    # Perform mutation on an offspring
    def mutate(offspring):
        for i in range(num_generators):
            if random.random() < mutation_rate:
                offspring[i] = random.uniform(p_min[i], p_max[i])
        return offspring

    # Main program
    num_generators = int(input("Enter the number of generators: "))
    p_min = []
    p_max = []
    a = []
    b = []
    c = []

    for i in range(num_generators):
        p_min.append(float(input(f"Enter the minimum power output of generator {i+1}: ")))
        p_max.append(float(input(f"Enter the maximum power output of generator {i+1}: ")))
        a.append(float(input(f"Enter the coefficient 'a' for generator {i+1}: ")))
        b.append(float(input(f"Enter the coefficient 'b' for generator {i+1}: ")))
        c.append(float(input(f"Enter the coefficient 'c' for generator {i+1}: ")))

    demand = float(input("Enter the total power demand: "))
    population_size = int(input("Enter the population size: "))
    max_iterations = int(input("Enter the maximum number of iterations: "))
    mutation_rate = float(input("Enter the mutation rate: "))

    population = initialize_population()

    best_solutions = []
    total_costs = []

    for iteration in range(max_iterations):
        fitness_values = evaluate_fitness(population)
        best_solution = population[np.argmin(fitness_values)]

        new_population = []
        new_population.append(best_solution)

        while len(new_population) < population_size:
            parent1 = select_parent(population, fitness_values)
            parent2 = select_parent(population, fitness_values)
            offspring = crossover(parent1, parent2)
            offspring = mutate(offspring)
            new_population.append(offspring)

        population = new_population

        best_solution = population[np.argmin(evaluate_fitness(population))]
        total_cost = calculate_total_cost(best_solution)

        best_solutions.append(best_solution)
        total_costs.append(total_cost)

        # Export current iteration results to CSV file
        results = [iteration + 1, best_solution, total_cost]
        headers = ["Iteration", "Best Solution", "Total Cost"]

        with open("eld_results.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if iteration == 0:
                writer.writerow(headers)
            writer.writerow(results)

    return best_solutions, total_costs

# Example usage
best_solutions, total_costs = cheetah_optimization()

print("Results exported to eld_results.csv")
