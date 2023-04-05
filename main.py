import random
import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
team_members = 10
project_hours = 120
max_hours_per_day = 5

# GA parameters
population_size = 50
generations = 10000
crossover_prob = 0.8
mutation_prob = 0.2

def create_individual():
    return [random.randint(0, max_hours_per_day) for _ in range(team_members)]

def evaluate(individual):
    total_hours = sum(individual)
    if total_hours == 0:
        return float("inf"),
    days = np.ceil(project_hours / total_hours)
    return days,

def selection(population, fitnesses, k):
    selected = []
    for _ in range(k):
        candidates = random.sample(range(len(population)), 3)
        selected.append(min(candidates, key=lambda x: fitnesses[x]))
    return [population[i] for i in selected]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.randint(0, max_hours_per_day)
    return individual,

def ga():
    # Create initial population
    population = [create_individual() for _ in range(population_size)]
    fitnesses = list(map(evaluate, population))

    # Main GA loop
    best_individual = None
    best_fitness = None
    for gen in range(generations):
        offspring = []

        # Apply crossover and mutation
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population, fitnesses, 2)
            if random.random() < crossover_prob:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            mutate(child1, 1 / team_members)
            mutate(child2, 1 / team_members)
            offspring.extend([child1, child2])

        # Update population
        population = offspring
        fitnesses = list(map(evaluate, population))

        # Record best individual
        current_best = min(population, key=evaluate)
        current_best_fitness = evaluate(current_best)[0]
        if best_fitness is None or current_best_fitness < best_fitness:
            best_individual = current_best
            best_fitness = current_best_fitness

    return best_individual, best_fitness

best_individual, best_fitness = ga()
print("Best individual:", best_individual)
print("Project duration (days):", best_fitness)

# Visualize the result
x = np.arange(team_members)
plt.bar(x, best_individual)
plt.xlabel("Team Member")
plt.ylabel("Hours Worked")
plt.title(f"Task Scheduling (Project duration: {best_fitness} days)")
plt.xticks(x)
plt.show()
