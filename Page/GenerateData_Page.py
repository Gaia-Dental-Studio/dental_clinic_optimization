import pandas as pd
from datetime import datetime
import logging
import streamlit as st
import openpyxl
from Dummy_scenario_generator import generate_dummy_scenario
from Treatment_dummy import *
from Scenario_Forecasting import *

import locale
locale.setlocale(locale.LC_ALL, '')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%message')


# Streamlit app
def display():
    # Load the necessary files
    clinic_labor_file = './Data/clinic_labor.xlsx'

    if 'dummy_df' not in st.session_state:
        st.session_state.dummy_df = pd.DataFrame()
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
    if 'clinic_labor_df' not in st.session_state:
        clinic_labor_df = pd.read_excel(clinic_labor_file, sheet_name='Dentists')
        st.session_state.clinic_labor_df = clinic_labor_df.copy()
    else:
        clinic_labor_df = st.session_state.clinic_labor_df

    st.title("Generate Dummy Page")

    st.header("Edit Dentist Roster")
    edited_df = st.data_editor(st.session_state.clinic_labor_df, num_rows="dynamic")

    if st.button("Save Changes"):
        st.session_state.clinic_labor_df = edited_df

        try:
            with pd.ExcelWriter(clinic_labor_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                edited_df.to_excel(writer, sheet_name='Dentists', index=False)
                
            st.success("Changes saved!")
        except Exception as e:
            st.error(f"An error occurred while saving changes: {e}")


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

   