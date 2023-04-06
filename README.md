ProjectManagement with GA
Python Code for Task Scheduling Optimization with Genetic Algorithm

Description
This Python code implements a genetic algorithm to optimize the scheduling of tasks assigned to a team of workers in order to minimize project duration and stay within a given budget. The problem considers the skills of team members, dependencies between tasks, and the risk of workers getting sick.

The code uses NumPy and Matplotlib libraries for data manipulation and visualization.

Usage
Download or clone the repository to your local machine.

Open the task_scheduling.py file in your preferred Python IDE or text editor.

Modify the problem parameters in the code to customize the problem to your specific requirements (see the parameters section below for details).

Run the code to execute the program.

The program will output the best individual, project duration, and additional budget required to complete the project.

A bar chart will also be generated showing the hours worked by each team member on each task.

Parameters
The following parameters can be modified in the code to customize the problem:

team_members: The number of workers in the team.
project_tasks: A list of dictionaries defining the tasks to be completed, including the hours required, skills required, and dependencies on other tasks.
team_member_skills: A list of dictionaries defining the skills of each team member.
max_hours_per_day: The maximum number of hours a worker can work in a day.
budget: The total budget allocated for the project.
normal_rate: The hourly rate for regular working hours.
overtime_rate: The hourly rate for overtime working hours.
overtime_threshold: The number of hours after which regular working hours become overtime working hours.
population_size: The size of the population in the genetic algorithm.
crossover_prob: The probability of crossover in the genetic algorithm.
generations: The number of generations in the genetic algorithm.
max_generations_without_improvement: The maximum number of generations without improvement before the genetic algorithm stops.
sick_probability: The probability of a worker getting sick on any given day.
sick_days: The number of days a worker is sick if they get sick.
License
