import numpy as np
import matplotlib.pyplot as plt

# Define problem parameters
num_tasks = 10
num_members = 5
workload_capacities = np.array([50, 60, 70, 80, 90])
task_difficulties = np.random.randint(1, 11, num_tasks)

# Define genetic algorithm parameters
pop_size = 50
num_generations = 100
mutation_rate = 0.1
tournament_size = 3
w1 = 0.5
w2 = 0.5

# Define fitness function
def fitness_function(solution):
    total_workload = np.sum(solution * task_difficulties)
    ideal_workload = np.sum(task_difficulties) / num_members
    project_duration = np.max(np.dot(solution, task_difficulties))
    min_duration = np.min(np.dot(population, task_difficulties))
    max_duration = np.max(np.dot(population, task_difficulties))
    workload_fitness = w1 * (1 - abs(total_workload - ideal_workload) / ideal_workload)
    duration_fitness = w2 * (1 - (project_duration - min_duration) / (max_duration - min_duration))
    fitness = workload_fitness + duration_fitness
    return fitness

# Generate initial population
population = np.random.randint(2, size=(pop_size, num_tasks))

# Keep track of best solution and its fitness over generations
best_solution = None
best_fitness = -np.inf
fitness_history = []

# Run genetic algorithm
for i in range(num_generations):
    # Evaluate fitness of population
    fitness_values = np.array([fitness_function(solution) for solution in population])

    # Update best solution and its fitness
    best_index = np.argmax(fitness_values)
    if fitness_values[best_index] > best_fitness:
        best_solution = population[best_index]
        best_fitness = fitness_values[best_index]

    # Add best fitness to history
    fitness_history.append(best_fitness)

    # Select parents using tournament selection
    parents = []
    for j in range(pop_size):
        tournament = np.random.choice(pop_size, tournament_size, replace=False)
        tournament_fitness = fitness_values[tournament]
        winner_index = tournament[np.argmax(tournament_fitness)]
        parents.append(population[winner_index])

    # Create offspring using single-point crossover
    offspring = []
    for j in range(pop_size):
        parent1 = parents[np.random.randint(len(parents))]
        parent2 = parents[np.random.randint(len(parents))]
        crossover_point = np.random.randint(num_tasks)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        offspring.append(child)

    # Mutate offspring
    for j in range(pop_size):
        for k in range(num_tasks):
            if np.random.rand() < mutation_rate:
                offspring[j][k] = 1 - offspring[j][k]

    # Evaluate fitness of offspring
    offspring_fitness = np.array([fitness_function(solution) for solution in offspring])

    # Select survivors using elitism selection
    sorted_indices = np.argsort(np.concatenate((fitness_values, offspring_fitness)))[::-1]
    population = np.array([np.copy(np.concatenate((population[index], offspring[index]))) for index in sorted_indices[:pop_size]])

# Print best solution and its fitness
print('Best solution:', best_solution)
print('Best fitness:', best_fitness)

# Plot fitness over generations
plt.plot(fitness_history)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
