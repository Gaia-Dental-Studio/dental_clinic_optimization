from flask import Flask, request, jsonify, send_file
import pandas as pd
import traceback
import matplotlib.pyplot as plt
import matplotlib
import json

# Use the 'Agg' backend for matplotlib
matplotlib.use('Agg')


from Page_Function import Generate_Train, Schedule_Optimization, Performance_Evaluation, Profit_Optimization, Run_Sheet_Optimization
from Page_Function.Classes import Financial_Model, Optimization_Functions

app = Flask(__name__)

@app.route('/forecast_scenario', methods=['POST'])
def post_forecast_scenario():
    try:
        # Check if the post request has the files part
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file1 = request.files['file1']
        file2 = request.files['file2']

        # Read the files into dataframes
        scenario_df = pd.read_csv(file1)
        service_df = pd.read_excel(file2)

        forecasted_df, model_evaluation, forecasted_profit = Generate_Train.train_and_forecast(scenario_df, service_df) 

        # Convert forecasted dataframe to a dictionary
        forecasted_data = forecasted_df.to_dict(orient='records')

        # Return the data as JSON response
        return jsonify({
            'forecasted_data': forecasted_data,
            'model_evaluation': model_evaluation,
            'forecasted_profit': forecasted_profit
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/optimize_schedule', methods=['POST'])
def post_optimize_schedule():
    try:
        # Check if the post request has the files part
        if 'file1' not in request.files or 'file2' not in request.files or 'file3' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        forecasted_df = pd.read_excel(request.files['file1'])
        dentists_df = pd.read_excel(request.files['file2'])
        selected_df = pd.read_excel(request.files['file3'])

        max_hours_per_week = int(request.form['max_hours_per_week'])
        max_hours_per_day = int(request.form['max_hours_per_day'])
 

        schedule_result = Schedule_Optimization.run_scheduling_optimization(max_hours_per_week, max_hours_per_day, dentists_df, forecasted_df, selected_df)

        # Return the data as JSON response
        return jsonify(schedule_result)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/performance_evaluation', methods=['POST'])
def post_dentist_performance():
    try:
        # Check if the post request has the files part
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        history_data = pd.read_excel(request.files['file1'])
        service_data = pd.read_excel(request.files['file2'])

        dentist_name = request.form['dentist_name']
        treatment_name = request.form['treatment_name']

        (
            treatment_count, 
            average_result, 
            max_result, 
            min_result, 
            std_result, 
            efficiency_result, 
            rating, 
            success_rate, 
            performance_score_result,
        ) = Performance_Evaluation.get_performance_metrics(
            dentist_name, 
            treatment_name, 
            history_data, 
            service_data, 
        )

        payload = {
           'dentist_name': dentist_name,
           'treatment_name': treatment_name,
           'treatment_counts': int(treatment_count),
           'average_count': float(average_result),
           'maximum_duration': float(max_result),
           'minimum_duration': float(min_result),
           'standard_deviation': float(std_result),
           'efficiency': float(efficiency_result),
           'treatment_rating': float(rating),
           'success_rate': float(format((success_rate * 100), '.3f')),
           'performance_score': performance_score_result
        }

        return payload

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/performance_evaluation_chart', methods=['POST'])
def post_dentist_performance_chart():
    try:
        # Check if the post request has the files part
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        history_data = pd.read_excel(request.files['file1'])
        service_data = pd.read_excel(request.files['file2'])

        dentist_name = request.form['dentist_name']
        treatment_name = request.form['treatment_name']

        (
            treatment_series, 
            target_duration, 
            start_month, 
            end_month
        ) = Performance_Evaluation.get_performance_metrics_chart(
            dentist_name, 
            treatment_name, 
            history_data, 
            service_data, 
        )
      
        plt.figure(figsize=(10, 6))
        plt.plot(treatment_series, marker='o', linestyle='-', label='Treatment Duration')
        plt.axhline(y=target_duration, color='r', linestyle='--', label=f'Expected Duration ({target_duration})')
        plt.title(f"{dentist_name} - {treatment_name}, History Trend")
        plt.xlabel('Date')
        plt.ylabel('Duration (minutes)')
        plt.legend()
        plt.grid(True)
        plot_filename = f"{dentist_name}-{treatment_name}-Performance-{start_month}-{end_month}.png"
        plt.savefig(plot_filename)
        plt.close()

        return send_file(plot_filename, mimetype='image/png')

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
@app.route('/profit_optimization', methods=['POST'])
def post_profit_optimization():
    try:
        # Check if the post request has the files part
        if 'file1' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        treatment_data = pd.read_excel(request.files['file1'])
        
        # Parse the JSON string from form data
        cost_payload = json.loads(request.form['cost_payload'])
        worker_payload = cost_payload.get('worker_payload')

        output_payload = Profit_Optimization.run_profit_optimization(cost_payload, worker_payload, treatment_data, Financial_Model, Optimization_Functions)

        return jsonify(output_payload)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
@app.route('/analyze_dentist_history', methods=['POST'])
def post_analyze_dentist_history():
    try:
        # Check if the post request has the files part
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        history_data = pd.read_excel(request.files['file1'])
        service_data = pd.read_excel(request.files['file2'])

        payload1, payload2, payload3, payload4 = Performance_Evaluation.Anlyze_dentist_past_data(history_data, service_data)

        # Combine the payloads into one JSON response
        return jsonify({
            "payload1": payload1,
            "payload2": payload2,
            "payload3": payload3,
            "payload4": payload4
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
@app.route('/run_sheet_optmization_v2-2', methods=['POST'])
def post_run_sheet_optimization():
    try:
        # Check if the post request has the files part
        if 'file1' not in request.files or 'file2' not in request.files or 'file3' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        constraint_data = pd.read_excel(request.files['file1'])
        worker_data = pd.read_excel(request.files['file2'])

        # Since 'file3' is a CSV file, use pd.read_csv
        forecasted_data = pd.read_csv(request.files['file3'])

        with open("./treatment_precedence_au.json", "r") as file:
            item_numbers_json = json.load(file)

        payload = Run_Sheet_Optimization.run_sheet_optimization(constraint_data, worker_data, forecasted_data, item_numbers_json)

        # Combine the payloads into one JSON response
        return payload

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
