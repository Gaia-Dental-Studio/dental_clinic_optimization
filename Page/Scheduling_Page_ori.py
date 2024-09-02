import pandas as pd
import random
from deap import base, creator, tools, algorithms
import numpy as np
from datetime import datetime
import logging
import os
import streamlit as st
from Dummy_scenario_generator import generate_dummy_scenario
from Treatment_dummy import *
from Scenario_Forecasting import *

import locale
locale.setlocale(locale.LC_ALL, '')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%message')

# Load the necessary files
clinic_labor_file = './Data/clinic_labor.xlsx'
service_list_file = './Data/Service_list.xlsx'

if 'clinic_labor_df' not in st.session_state:
    clinic_labor_df = pd.read_excel(clinic_labor_file, sheet_name='Artificial')
    st.session_state.clinic_labor_df = clinic_labor_df.copy()
else:
    clinic_labor_df = st.session_state.clinic_labor_df

service_list_df = pd.read_excel(service_list_file, sheet_name='Main')

# Convert to dictionary for quick lookup
treatment_durations = dict(zip(service_list_df['Treatment Menu Names'], service_list_df['Duration']))

# Function to get adjusted duration based on the dentist's specialty
def get_adjusted_duration(treatment, dentist):
    if treatment == dentist['Specialty']:
        return treatment_durations[treatment] * (1 - dentist['Efficiency'])
    else:
        return treatment_durations[treatment]

def create_individual(hour_required, max_hours_per_day, max_hours_per_week):
    sorted_dentists = sorted(dentists, key=lambda x: x['Wage_per_hour'])
    individual = []

    for total_hours in hour_required:
        daily_hours = [0] * len(dentists)
        remaining_hours = total_hours

        for dentist in sorted_dentists:
            if remaining_hours <= 0:
                break

            max_hours_available = min(max_hours_per_day, max_hours_per_week - sum(daily_hours))
            hours_to_assign = min(remaining_hours, max_hours_available)

            daily_hours[dentists.index(dentist)] = hours_to_assign
            remaining_hours -= hours_to_assign

        individual.extend(daily_hours)

    return individual

# Evaluate the cost of an individual
def evaluate_individual(individual, hour_required, max_hours_per_day, max_hours_per_week):
    total_cost = 0
    daily_hours = {dentist['Name']: 0 for dentist in dentists}
    weekly_hours = {dentist['Name']: 0 for dentist in dentists}

    penalty = 0
    for i in range(len(hour_required)):
        for j in range(len(dentists)):
            hours = individual[i * len(dentists) + j]
            if hours > 0:
                daily_hours[dentists[j]['Name']] += hours
                weekly_hours[dentists[j]['Name']] += hours
                total_cost += hours * dentists[j]['Wage_per_hour']

        # Check total hours required with efficiency adjustment
        total_hours_needed = hour_required[i]
        total_hours_available = sum(daily_hours.values())
        max_hours_with_efficiency = total_hours_needed * (1 - 0.15)  # 15% average efficiency
        if total_hours_available < max_hours_with_efficiency:
            penalty += (max_hours_with_efficiency - total_hours_available) * 100000  # Stronger penalty for unmet hours

        # Check daily constraints
        if any(hours > max_hours_per_day for hours in daily_hours.values()):
            return (float('inf'),)

    # Check weekly constraints
    if any(hours > max_hours_per_week for hours in weekly_hours.values()):
        return (float('inf'),)

    return (penalty + total_cost * 0.1,)  # Lower penalty for cost

# Create fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Genetic Algorithm Setup
def genetic_algorithm_optimization(hour_required, daily_demands, max_hours_per_day, max_hours_per_week, tolerance=0.05, max_stagnation=10):
    random.seed(64)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual(hour_required, max_hours_per_day, max_hours_per_week))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: evaluate_individual(ind, hour_required, max_hours_per_day, max_hours_per_week))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=2000)  # Increased population size
    hof = tools.HallOfFame(20)  # Store more individuals
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Convergence stopping condition
    prev_best_fitness = None
    stagnation_counter = 0
    final_generation = 0  # To keep track of the number of generations

    for gen in range(100):  # Run for a fixed number of generations
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=1, stats=stats, halloffame=hof, verbose=False)
        final_generation = gen + 1  # Update the final generation number

        # Get the best fitness in the current population
        current_best_fitness = logbook.select("min")[-1]

        if prev_best_fitness is None or current_best_fitness < prev_best_fitness:
            prev_best_fitness = current_best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Stop if the population is stagnating
        if stagnation_counter >= max_stagnation:
            break

    top_individuals = hof
    top_schedules = []

    for best_individual in top_individuals:  # Iterate over top individuals to find valid schedules
        schedule_summary = []
        valid_schedule = True

        for i, date in enumerate(daily_demands.keys()):
            hours_worked = {dentist['Name']: 0 for dentist in dentists}
            total_cost = 0
            total_hours_available = 0

            for j, dentist in enumerate(dentists):
                hours = best_individual[i * len(dentists) + j]
                if hours > 0:
                    hours_worked[dentist['Name']] += hours
                    total_cost += hours * dentist['Wage_per_hour']
                    total_hours_available += hours

            # Filter out dentists working less than an hour
            hours_worked = {k: v for k, v in hours_worked.items() if v >= 1}

            # Check if the required hours are met within tolerance
            total_hours_needed = hour_required[i]
            if total_hours_available < (total_hours_needed * (1 - tolerance)) or total_hours_available > (total_hours_needed * (1 + tolerance)):
                valid_schedule = False
                break

            schedule_summary.append({
                'Date': date,
                'Dentist Name': list(hours_worked.keys()),
                'Working hours': [round(v, 3) for v in hours_worked.values()],
                'Total Cost': round(total_cost, 3)
            })

        if valid_schedule:
            top_schedules.append(schedule_summary)

        # Stop after finding 3 valid schedules
        if len(top_schedules) >= 3:
            break

    return top_schedules, final_generation

# Streamlit app
def display():
    if 'dummy_df' not in st.session_state:
        st.session_state.dummy_df = pd.DataFrame()
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
    if 'clinic_labor_df' not in st.session_state:
        clinic_labor_df = pd.read_excel(clinic_labor_file, sheet_name='Artificial')
        st.session_state.clinic_labor_df = clinic_labor_df.copy()
    else:
        clinic_labor_df = st.session_state.clinic_labor_df

    st.title("Scheduling Optimization")

    max_hours_per_week = st.number_input("Maximum Working Hours per Week", value=40, min_value=1)
    max_hours_per_day = st.number_input("Maximum Working Hours per Day", value=10, min_value=1)

    st.header("Edit Dentist Roster")
    edited_df = st.data_editor(st.session_state.clinic_labor_df, num_rows="dynamic")

    if st.button("Save Changes"):
        st.session_state.clinic_labor_df = edited_df
        st.success("Changes saved!")

    # Extract relevant columns from the updated clinic_labor_df
    clinic_labor_processed = edited_df[['Dentist Name', 'Specialty', 'Efficiency', 'Wage/Hour']]
    clinic_labor_processed.columns = ['Name', 'Specialty', 'Efficiency', 'Wage_per_hour']
    
    # Prepare data for optimization
    global dentists, services, durations
    dentists = clinic_labor_processed.to_dict(orient='records')
    services = service_list_df['Treatment Menu Names'].tolist()
    durations = service_list_df['Duration'].tolist()

    st.header("Dummy Scenario Generator")
    start_date = st.date_input("Start Date", value=datetime(2023, 6, 5))
    end_date = st.date_input("End Date", value=datetime(2023, 6, 9))
    random_maximum = st.number_input("Maximum Treatment Count per Day", value=2, min_value=1)

    if st.button("Generate Dummy Scenario & Forecast"):
        if start_date > end_date:
            st.error("Start Date should be before End Date")
        else:
            dummy_df = generate_dummy_scenario(start_date, end_date, random_maximum)
            create_dummy_treatments()
            train_response = train_and_forecast()
            st.session_state.dummy_df = dummy_df
            st.success(f"Dummy Scenario generated! & {train_response}")

    # Display the dummy dataframe if it exists
    if not st.session_state.dummy_df.empty:
        st.write(st.session_state.dummy_df)

    if st.button("Optimize Schedule"):
        if 'dummy_df' not in st.session_state:
            st.error("Please generate dummy scenario first!")
        else:
            dummy_df = st.session_state.dummy_df
            weekly_demand = dummy_df.to_dict(orient='records')
            daily_demands = {row['Date']: {service: row[service] for service in services if service in row} for row in weekly_demand}
            hour_required = dummy_df['Hour Required'].tolist()

            top_schedules, final_generation = genetic_algorithm_optimization(hour_required, daily_demands, max_hours_per_day, max_hours_per_week)

            st.write(f"Algorithm stopped after {final_generation} iterations.")

            for idx, best_schedules in enumerate(top_schedules, start=1):
                st.write(f"Roster {idx} Schedules:")
                results_df = pd.DataFrame(best_schedules)
                st.write(results_df)
                total_cost_sum = results_df['Total Cost'].sum()
                st.write(f"Total Cost Sum for Roster {idx}: {locale.format_string('%d', total_cost_sum , grouping=True)} IDR")

                # Save the results
                output_dir = './Results/'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_file = os.path.join(output_dir, f'optimal_schedule_top_{idx}.xlsx')
                results_df.to_excel(output_file, index=False)

                st.download_button(
                    label=f"Download Roster {idx} Optimal Schedules as Excel",
                    data=results_df.to_csv(index=False).encode('utf-8'),
                    file_name=f'optimal_schedules_top_{idx}.csv',
                    mime='text/csv'
                )