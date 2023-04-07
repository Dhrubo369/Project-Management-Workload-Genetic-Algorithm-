import random
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
team_members = 5
project_tasks = [
    {"id": 0, "hours": 10, "required_project_skills": [0], "dependencies": []},
    {"id": 1, "hours": 10, "required_project_skills": [1], "dependencies": [0]},
    {"id": 2, "hours": 10, "required_project_skills": [0, 1], "dependencies": [1]},
    {"id": 3, "hours": 10, "required_project_skills": [2], "dependencies": [2]},
    {"id": 4, "hours": 10, "required_project_skills": [3, 4], "dependencies": [2, 3]},
]

team_member_skills = [
    {"id": 0, "skills": [0, 1]},
    {"id": 1, "skills": [1]},
    {"id": 2, "skills": [2,3]},
    {"id": 3, "skills": [3,1]},
    {"id": 4, "skills": [4,1,2,3]},
]

max_hours_per_day = 5

# Budget parameters
budget = 500
normal_rate = 20
overtime_rate = 30
overtime_threshold = 4

# GA parameters
population_size = 10000
crossover_prob = 0.5
generations = 10000
max_generations_without_improvement = 10

# Risk management parameters
sick_probability = 0.1
sick_days = 2

def is_member_available(member_availability, task_dependencies):
    for dep_id in task_dependencies:
        if member_availability[dep_id] > 0:
            return False
    return True

def find_next_available_day(member_availability, task_dependencies):
    max_dependency_end_day = 0
    for dep_id in task_dependencies:
        max_dependency_end_day = max(max_dependency_end_day, member_availability[dep_id])
    return max_dependency_end_day

def create_individual():
    individual = [0 for x in range(team_members * len(project_tasks))]
    member_availability = [[0 for x in range(len(project_tasks))] for y in range(team_members)]

    for task in project_tasks:
        capable_members = [
            member for member in team_member_skills
            if can_member_perform_task(member["skills"], task["required_project_skills"])
        ]

        if not capable_members:
            continue

        random.shuffle(capable_members)  # Shuffle the list of capable members

        for member in capable_members:
            assigned_member = member["id"]
            task_index = assigned_member * len(project_tasks) + task["id"]

            if not is_member_available(member_availability[assigned_member], task["dependencies"]):
                continue

            assigned_hours = random.randint(1, max_hours_per_day)
            required_days = int(np.ceil(task["hours"] / assigned_hours))

            next_available_day = find_next_available_day(member_availability[assigned_member], task["dependencies"])
            member_availability[assigned_member][task["id"]] = next_available_day + required_days

            individual[task_index] = assigned_hours
            break
    return individual




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
    sick_hours = np.random.binomial(sick_days, sick_probability, team_members * len(project_tasks))
    adjusted_individual = np.maximum(np.array(individual) - sick_hours, 0)

    # Calculate the time required to complete each task
    task_completion_times = []
    for task in project_tasks:
        task_times = []
        for i in range(team_members):
            index = i * len(project_tasks) + task["id"]
            hours = adjusted_individual[index]
            if can_member_perform_task(team_member_skills[i]["skills"], task["required_project_skills"]) and hours > 0:
                task_times.append(calculate_task_completion_time(task["hours"], hours))
        if not task_times:
            task_times.append(calculate_task_completion_time(task["hours"], max_hours_per_day))

        task_completion_times.append(min(task_times))

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
    selected = random.choices(range(len(population)), weights=[1 / fit[0] for fit in fitnesses], k=k)
    return [population[i] for i in selected]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            team_member_id = i // len(project_tasks)
            task_id = i % len(project_tasks)
            member_skills = team_member_skills[team_member_id]["skills"]
            task_skills = project_tasks[task_id]["required_project_skills"]

            if can_member_perform_task(member_skills, task_skills):
                individual[i] = random.randint(1, max_hours_per_day)
            else:
                individual[i] = 0
    return individual,


def ga():
    # Create initial population
    population = [create_individual() for _ in range(population_size)]
    fitnesses = list(map(evaluate, population))

    # Main GA loop
    best_individual = None
    best_fitness = float("inf")
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

            mutate(child1, 1 / len(project_tasks))  
            mutate(child2, 1 / len(project_tasks))
            offspring.extend([child1, child2])

        # Update population
        population = offspring
        fitnesses = list(map(evaluate, population))

        # Track best individual
        min_index, current_best_fitness = min(enumerate(fitnesses), key=lambda x: x[1][0])
        current_best = population[min_index]
        if current_best_fitness[0] < best_fitness:
            best_individual = current_best
            best_fitness = current_best_fitness[0]
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Check stopping criterion
        if generations_without_improvement >= max_generations_without_improvement:
            break

    return best_individual, best_fitness

def calculate_task_times(best_individual):
    sick_hours = np.random.binomial(sick_days, sick_probability, team_members * len(project_tasks))
    adjusted_individual = np.maximum(np.array(best_individual) - sick_hours, 0)

    task_times = []
    for task in project_tasks:
        task_start_time = 0
        for dep_id in task["dependencies"]:
            task_start_time = max(task_start_time, task_times[dep_id]["end_time"])

        task_assigned_hours = 0
        for i in range(team_members):
            index = i * len(project_tasks) + task["id"]
            hours = adjusted_individual[index]
            if hours > 0:
                task_assigned_hours = hours
                break

        if task_assigned_hours > 0:
            task_duration = calculate_task_completion_time(task["hours"], task_assigned_hours)
            task_end_time = task_start_time + task_duration
            task_times.append({"start_time": task_start_time, "end_time": task_end_time})
        else:
            task_times.append({"start_time": float("inf"), "end_time": float("inf")})
    
    return task_times



def show_schedule(best_individual):
    root = tk.Tk()
    root.title("Task Scheduling")

    header_font = ("Arial", 12, "bold")
    data_font = ("Arial", 10)

    # Create header row
    tk.Label(root, text="Team Member / Task", font=header_font).grid(row=0, column=0)

    for j, task in enumerate(project_tasks):
        task_label = f"Task {task['id']}"
        tk.Label(root, text=task_label, font=header_font, bg="lightblue").grid(row=0, column=j + 1)

    # Fill data rows
    for i, member in enumerate(team_member_skills):
        member_label = f"Team Member {i}"
        tk.Label(root, text=member_label, font=header_font, bg="lightblue").grid(row=i + 1, column=0)

        for j, task in enumerate(project_tasks):
            index = i * len(project_tasks) + task["id"]
            hours = best_individual[index]

            if hours > 0:
                cell_text = f"{hours}"
                cell_bg = "lightgreen" if can_member_perform_task(member["skills"], task["required_project_skills"]) else "orange"
            else:
                cell_text = ""
                cell_bg = None

            tk.Label(root, text=cell_text, font=data_font, bg=cell_bg).grid(row=i + 1, column=j + 1)

    root.mainloop()

def plot_gantt_chart(task_times, project_duration):
    gantt_chart_duration = max(times["end_time"] for times in task_times if not np.isnan(times["end_time"]) and not np.isinf(times["end_time"]))
    scaling_factor = project_duration / gantt_chart_duration

    fig, ax = plt.subplots()

    for task, times in enumerate(task_times):
        if not np.isnan(times["start_time"]) and not np.isinf(times["start_time"]) and not np.isnan(times["end_time"]) and not np.isinf(times["end_time"]) and not np.isnan(scaling_factor) and not np.isinf(scaling_factor):
            start_time = times["start_time"] * scaling_factor
            end_time = times["end_time"] * scaling_factor
            duration = end_time - start_time
            ax.barh(task, duration, left=start_time)


    ax.set_yticks(range(len(project_tasks)))
    ax.set_yticklabels([f"Task {task['id']}" for task in project_tasks])
    ax.set_xlabel("Days")
    ax.set_ylabel("Tasks")
    ax.set_title(f"Gantt Chart (Project duration: {project_duration} days)")

    plt.show()




best_individual, best_fitness = ga()
total_duration, budget_difference = evaluate(best_individual)
print("Best individual:", best_individual)
print("Project duration (days):", total_duration)
print("Additional budget required:", budget_difference)

show_schedule(best_individual)
task_times = calculate_task_times(best_individual)
plot_gantt_chart(task_times, total_duration)
x = np.arange(team_members * len(project_tasks))
plt.bar(x, best_individual)
plt.xlabel("Team Member * Task ID")
plt.ylabel("Hours Worked")
plt.title(f"Task Scheduling (Project duration: {total_duration} days)")
plt.xticks(x)
plt.show()