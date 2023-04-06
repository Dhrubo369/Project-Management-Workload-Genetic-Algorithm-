import random
import numpy as np
import matplotlib.pyplot as plt

import random
import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
team_members = 5
project_work_units = 600
max_hours_per_day = 5

# Team productivity (work units per hour)
team_productivity = [10, 12]

# Budget parameters
budget = 100
normal_rate = 50
overtime_rate = 60
overtime_threshold = 4

# GA parameters
population_size = 100
generations = 100000
crossover_prob = 0.8
max_generations_without_improvement = 100

# Risk management parameters
working_days = 10
sick_days = 2

# Milestone parameters
milestone_work_units = 200

def create_individual():
    return [random.randint(0, max_hours_per_day) for _ in range(team_members)]

def calculate_cost(individual):
    hours = np.array(individual)
    normal_hours = np.minimum(hours, overtime_threshold)
    overtime_hours = np.maximum(hours - overtime_threshold, 0)
    total_cost = np.sum(normal_hours * normal_rate + overtime_hours * overtime_rate)
    return total_cost

def evaluate(individual):
    total_hours = sum(individual)
    if total_hours == 0:
        return float("inf"), float("inf")
    
    # Calculate sick probability for each team member
    sick_probabilities = [1 - ((working_days - sick_days) / working_days)**total_hours for total_hours in individual]
    
    # Account for the risk of team members being sick
    sick_hours = np.random.binomial(sick_days, sick_probabilities)
    adjusted_individual = np.maximum(np.array(individual) - sick_hours, 0)
    adjusted_work_units = np.sum(np.array(adjusted_individual) * np.array(team_productivity))
    
    days = np.ceil(project_work_units / adjusted_work_units)
    total_cost = calculate_cost(individual)
    
    budget_difference = 0
    if total_cost > budget:
        budget_difference = total_cost - budget
        penalty = budget_difference / (budget * 0.01)  # penalty as a percentage of the budget
        days += penalty
        
    # Check if each team member meets the milestone requirement
    individual_work_units = np.array(individual) * np.array(team_productivity)
    for work_units in individual_work_units:
        if work_units < milestone_work_units:
            days += 1  # Add a penalty of 1 day if the team member does not meet the milestone requirement
    
    return days, budget_difference




def tournament_selection(population, fitnesses, k):
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
    generations_without_improvement = 0
    for gen in range(generations):
        offspring = []

        # Apply crossover and mutation
        for _ in range(population_size // 2):
            parent1, parent2 = tournament_selection(population, fitnesses, 2)
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

        # Track best individual
        min_index = np.argmin(fitnesses)
        current_best = population[min_index]
        current_best_fitness = fitnesses[min_index]
        if best_fitness is None or current_best_fitness < best_fitness:
            best_individual = current_best
            best_fitness = current_best_fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Check stopping criterion
        if generations_without_improvement >= max_generations_without_improvement:
            break

    return best_individual, best_fitness

best_individual, (best_fitness, budget_difference) = ga()
print("Best individual:", best_individual)
print("Project duration (days):", best_fitness)
print("Additional budget required:", budget_difference)

# Visualize the result
x = np.arange(team_members)
plt.bar(x, best_individual)
plt.xlabel("Team Member")
plt.ylabel("Hours Worked")
plt.title(f"Task Scheduling (Project duration: {best_fitness} days)")
plt.xticks(x)
plt.show()
