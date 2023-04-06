import random
import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
team_members = 2
project_tasks = [
    {"id": 0, "hours": 20, "skills_required": [0], "dependencies": []},
    {"id": 1, "hours": 30, "skills_required": [1], "dependencies": [0]},
    {"id": 2, "hours": 10, "skills_required": [0, 1], "dependencies": [1]},
]

team_member_skills = [
    {"id": 0, "skills": [0, 1]},
    {"id": 1, "skills": [1]},
]

max_hours_per_day = 5

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
sick_probability = 0.5
sick_days = 2

def create_individual():
    return [random.randint(0, max_hours_per_day) for _ in range(team_members)]

def calculate_cost(individual):
    hours = np.array(individual)
    normal_hours = np.minimum(hours, overtime_threshold)
    overtime_hours = np.maximum(hours - overtime_threshold, 0)
    total_cost = np.sum(normal_hours * normal_rate + overtime_hours * overtime_rate)
    return total_cost

def can_member_perform_task(member_skills, task_skills):
    return all(skill in member_skills for skill in task_skills)

def calculate_task_completion_time(task_hours, assigned_hours):
    return np.ceil(task_hours / assigned_hours)

def evaluate(individual):
    # Account for the risk of team members being sick
    sick_hours = np.random.binomial(sick_days, sick_probability, team_members)
    adjusted_individual = np.maximum(np.array(individual) - sick_hours, 0)
    
    # Calculate the time required to complete each task
    task_completion_times = []
    for task in project_tasks:
        task_times = []
        for i, hours in enumerate(adjusted_individual):
            if can_member_perform_task(team_member_skills[i]["skills"], task["skills_required"]):
                task_times.append(calculate_task_completion_time(task["hours"], hours))
        if task_times:
            task_completion_times.append(min(task_times))
        else:
            return float("inf"), float("inf")  # Task cannot be completed by any team member

    # Calculate project duration considering dependencies
    max_dependency_time = [0] * len(project_tasks)
    for task, completion_time in zip(project_tasks, task_completion_times):
        for dep_id in task["dependencies"]:
            max_dependency_time[task["id"]] = max(max_dependency_time[task["id"]], max_dependency_time[dep_id] + task_completion_times[dep_id])
        max_dependency_time[task["id"]] += completion_time

    project_duration = max(max_dependency_time)

    total_cost = calculate_cost(individual)

    budget_difference = 0
    if total_cost > budget:
        budget_difference = total_cost - budget
        penalty = budget_difference / (budget * 0.01)  # penalty as a percentage of the budget
        project_duration += penalty
    return project_duration, budget_difference

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
best_individual, (best_fitness, budget_difference) = ga()
print("Best individual:", best_individual)
print("Project duration (days):", best_fitness)
print("Additional budget required:", budget_difference)

x = np.arange(team_members)
plt.bar(x, best_individual)
plt.xlabel("Team Member")
plt.ylabel("Hours Worked")
plt.title(f"Task Scheduling (Project duration: {best_fitness} days)")
plt.xticks(x)
plt.show()