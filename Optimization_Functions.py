import numpy as np
from deap import base, creator, tools, algorithms

# GA Optimization class
class GAOptimization:
    def __init__(self, prices, durations, total_cost_per_clinic, total_working_hours_per_month, profits, cogs, service_counts=None, use_bias=False, cogs_limit=None, patient_limit=None, bias_weight=0.7):
        self.prices = prices
        self.durations = durations
        self.total_cost_per_clinic = total_cost_per_clinic
        self.total_working_hours_per_month = total_working_hours_per_month
        self.profits = profits
        self.cogs = cogs
        self.service_counts = service_counts
        self.use_bias = use_bias
        self.cogs_limit = cogs_limit
        self.patient_limit = patient_limit
        self.bias_weight = bias_weight  # Add bias weight

    def evaluate(self, individual):
        revenue = np.dot(self.prices, individual)
        cost = self.total_cost_per_clinic
        profit = revenue - cost
        total_time = np.dot(self.durations, individual)
        total_cogs = np.dot(self.cogs, individual)
        total_patients = sum(individual)

        if total_time > self.total_working_hours_per_month or (self.cogs_limit is not None and total_cogs > self.cogs_limit) or (self.patient_limit is not None and total_patients > self.patient_limit):
            return -1e6,  # Large penalty if constraints are violated
        return profit,

    def initialize_individual(self):
        individual = []
        remaining_hours = self.total_working_hours_per_month
        remaining_cogs = self.cogs_limit if self.cogs_limit is not None else 1e12  # Large value instead of float('inf')
        remaining_patients = self.patient_limit if self.patient_limit is not None else 1e12  # Large value instead of float('inf')
        
        if self.use_bias and self.patient_limit:
            max_service_count = max(self.service_counts.values())
            normalized_counts = [self.service_counts.get(service_name, 0) / max_service_count for service_name in self.service_counts]
            combined_bias = [self.bias_weight * count + (1 - self.bias_weight) for count in normalized_counts]
        elif self.use_bias:
            profit_per_hour = np.array(self.profits) / np.array(self.durations)
            max_service_count = max(self.service_counts.values())
            normalized_counts = [self.service_counts.get(service_name, 0) / max_service_count for service_name in self.service_counts]
            combined_bias = [self.bias_weight * count + (1 - self.bias_weight) * (profit / max(profit_per_hour)) for count, profit in zip(normalized_counts, profit_per_hour)]
        else:
            profit_per_hour = np.array(self.profits) / np.array(self.durations)
            combined_bias = [profit / max(profit_per_hour) for profit in profit_per_hour]

        for duration, bias, cogs in zip(self.durations, combined_bias, self.cogs):
            max_count_hours = int((remaining_hours / duration) * bias)
            max_count_cogs = int(remaining_cogs / cogs) if cogs > 0 else 1e12  # Large value instead of float('inf')
            max_count_patients = int(remaining_patients) if remaining_patients != 1e12 else 1e12  # Large value instead of float('inf')
            max_count = min(max_count_hours, max_count_cogs, max_count_patients)
            count = np.random.randint(0, max_count + 1)
            individual.append(count)
            remaining_hours -= count * duration
            remaining_cogs -= count * cogs
            remaining_patients -= count
            if remaining_hours < 0 or remaining_cogs < 0 or remaining_patients < 0:
                individual[-1] += int(min(remaining_hours / duration, remaining_cogs / cogs, remaining_patients))
                break

        while len(individual) < len(self.durations):
            individual.append(0)

        return creator.Individual(individual)
        

    def mutate_individual(self, individual):
        mutation_rate = 0.1
        intense_mutation_rate = 0.3
        intense_mutation_chance = 0.1

        remaining_cogs = self.cogs_limit if self.cogs_limit is not None else 1e12
        remaining_patients = self.patient_limit if self.patient_limit is not None else 1e12
        total_cogs = np.dot(self.cogs, individual)
        total_patients = sum(individual)

        if np.random.rand() < intense_mutation_chance:
            for i in range(len(individual)):
                if np.random.rand() < intense_mutation_rate:
                    change = np.random.randint(-3, 4)
                    new_value = max(0, individual[i] + change)
                    new_total_cogs = total_cogs - (self.cogs[i] * individual[i]) + (self.cogs[i] * new_value)
                    new_total_patients = total_patients - individual[i] + new_value
                    if new_total_cogs <= remaining_cogs and new_total_patients <= remaining_patients:
                        individual[i] = new_value
                        total_cogs = new_total_cogs
                        total_patients = new_total_patients
        else:
            for i in range(len(individual)):
                if np.random.rand() < mutation_rate:
                    change = np.random.randint(-1, 2)
                    new_value = max(0, individual[i] + change)
                    new_total_cogs = total_cogs - (self.cogs[i] * individual[i]) + (self.cogs[i] * new_value)
                    new_total_patients = total_patients - individual[i] + new_value
                    if new_total_cogs <= remaining_cogs and new_total_patients <= remaining_patients:
                        individual[i] = new_value
                        total_cogs = new_total_cogs
                        total_patients = new_total_patients

        return individual,

    def optimize(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("individual", self.initialize_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate_individual)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)

        population_size = 200
        num_generations = 200
        crossover_probability = 0.8
        mutation_probability = 0.3

        population = toolbox.population(n=population_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability, ngen=num_generations, stats=stats, verbose=False)
        
        best_individual = tools.selBest(population, k=1)[0]
        return best_individual, logbook
