import numpy as np
import pandas as pd

def input_clinic_cost(cost_payload, Financial_Model):

    cost_list = Financial_Model.CostList(cost_list=cost_payload)

    return cost_list

def input_clinic_treatments(treatment_payload, Financial_Model):

    treatments_list = Financial_Model.ClinicTreatmentList({"treatments":treatment_payload, "treatment_discount_rate": 0.0, "treatment_debit_charge": 0.0})

    return treatments_list


def run_profit_optimization(cost_payload, worker_payload, df_treatment, Financial_Model, Optimization_Functions):

    #Change Minutes to Hour
    df_treatment['Duration'] = df_treatment['Duration'] / 60
    df_treatment['Duration'] = df_treatment['Duration'].apply(lambda x: round(x, 3))  # Round to 3 decimal places


    total_working_hours_per_month = worker_payload['maximum_hour_per_week'] * worker_payload['dentist_count'] * 4
    salary_cost = worker_payload['dentist_count'] * worker_payload['average_salary']

    treatment_processed = []
    treatments = []

    for index, row in df_treatment.iterrows():
        if index not in treatment_processed:
            treatment = {
                'treatment_id': index,
                'treatment_name': row['Treatment Menu Names'],
                'treatment_fee': row['Price'],
                'treatment_cogs': row['COGS'],
                'treatment_duration': row['Duration'],
                'treatment_discount_rate': row['Discount Rate'],
                'treatment_debit_charge': row['Debit Charge']
            }
            treatments.append(treatment)
            treatment_processed.append(index)

    cost_list = input_clinic_cost(cost_payload, Financial_Model)
    treatment_list = input_clinic_treatments(treatments, Financial_Model)

    total_cost_per_clinic = cost_list.get_total_cost() + salary_cost
    prices = [treatment.get_treatment_fee() for treatment in treatment_list.get_treatments()]
    durations = [treatment.get_treatment_duration() for treatment in treatment_list.get_treatments()]
    COGS = [treatment.get_treatment_COGS() for treatment in treatment_list.get_treatments()]

    treatments = treatment_list.get_treatments()

    # Net profits calculation
    profits = [treatment.calculate_treatment_netProfit() for treatment in treatments]

    #Some Declared Variables
    use_bias = False
    treatment_count = None

    iteration = 0
    max_iterations = 20 
    efficiency = 0
    best_efficiency = 0
    best_individual = None
    best_logbook = None

    model_type = "General Model"
    constraint_type = "Optimization with Hour Limit"

    COGS_limit = None
    patient_limit = None

    # total_profit = 0

    # print('price: ', prices)
    # print('total_clinic_cost: ', total_cost_per_clinic)
    # print('total_working_hours: ', total_working_hours_per_month)
    # print('profit_each_treatment: ', profits)
    # print('COGS:', COGS)
    # print(service_counts)

    while iteration < max_iterations:
        ga_optimizer = Optimization_Functions.GAOptimization(prices, durations, total_cost_per_clinic, total_working_hours_per_month, profits, COGS,  treatment_count, use_bias, COGS_limit, patient_limit, bias_weight=0.9)
        current_individual, logbook = ga_optimizer.optimize()

        total_hours_used = np.dot(current_individual, durations)
        efficiency = (total_hours_used / total_working_hours_per_month) * 100

        if constraint_type == "Optimization with Hour Limit" and efficiency >= 10:
            best_individual = current_individual
            best_logbook = logbook
            break

        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_individual = current_individual
            best_logbook = logbook

        iteration += 1

    # Display the service counts for each service
    treatment_names = [treatment.get_treatment_name() for treatment in treatment_list.get_treatments()]

    print(best_individual)

    payload_output = {
        'model_information':{
            'best_individual': best_individual,
            'fitness_value': best_individual.fitness.values[0],
            'maximum_profit': sum(np.array(best_individual) * np.array(profits)),
            'total_hour': float(format(total_hours_used, '.3f')),
            'model_efficiency': float(format(efficiency, '.2f'))
        },
        'optimize_treatments':{
            'treatment_name': treatment_names,
            'optimize_count': best_individual,
            'total_hours': [count * duration for count, duration in zip(best_individual, durations)],
            'profit_contribution': [ count * profit for count, profit in zip(best_individual, profits)]
        }
    }

    return payload_output


