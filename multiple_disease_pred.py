import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Loading the saved models
base_path = r'C:\Computer Science Engineering\1. Important Projects\Multiple Disease Prediction system\saved models'
diabetes_model = pickle.load(open(os.path.join(base_path, 'diabetes_model.sav'), 'rb'))
heart_disease_model = pickle.load(open(os.path.join(base_path, 'heart_disease_model.sav'), 'rb'))
parkinsons_model = pickle.load(open(os.path.join(base_path, 'parkinsons_model.sav'), 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinsons Prediction"],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Function to safely convert input to float
def handle_invalid_input(value):
    try:
        return float(value)
    except ValueError:
        return None

# -------------------- Diabetes Prediction --------------------
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    
    with st.form("diabetes_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.text_input('Number of Pregnancies')
            SkinThickness = st.text_input('Skin Thickness value')
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        with col2:
            Glucose = st.text_input('Glucose Level')
            Insulin = st.text_input('Insulin Level')
            Age = st.text_input('Age of the Person')
        with col3:
            BloodPressure = st.text_input('Blood Pressure value')
            BMI = st.text_input('BMI value')

        submitted = st.form_submit_button('Get Diabetes Test Result')

        if submitted:
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            user_input = [handle_invalid_input(x) for x in user_input]
            
            if None in user_input:
                st.error('Please enter valid numerical inputs.')
            else:
                result = diabetes_model.predict([user_input])[0]
                st.success('The person is diabetic.' if result == 1 else 'The person is not diabetic.')

# -------------------- Heart Disease Prediction --------------------
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.text_input('Age')
            trestbps = st.text_input('Resting Blood Pressure')
            restecg = st.text_input('Resting Electrocardiographic results')
            oldpeak = st.text_input('ST depression induced by exercise')
            thal = st.text_input('Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)')
        with col2:
            sex = st.text_input('Sex (1 = male; 0 = female)')
            chol = st.text_input('Serum Cholestoral in mg/dl')
            thalach = st.text_input('Maximum Heart Rate achieved')
            slope = st.text_input('Slope of peak exercise ST segment')
        with col3:
            cp = st.text_input('Chest Pain types')
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)')
            exang = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)')
            ca = st.text_input('Number of major vessels colored by flourosopy')

        submitted = st.form_submit_button('Get Heart Disease Test Result')

        if submitted:
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]
            user_input = [handle_invalid_input(x) for x in user_input]

            if None in user_input:
                st.error('Please enter valid numerical inputs.')
            else:
                result = heart_disease_model.predict([user_input])[0]
                st.success('The person has heart disease.' if result == 1 else 'The person does not have heart disease.')

# -------------------- Parkinson's Prediction --------------------
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    with st.form("parkinsons_form"):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            fo = st.text_input('MDVP:Fo(Hz)')
            RAP = st.text_input('MDVP:RAP')
            APQ3 = st.text_input('Shimmer:APQ3')
            HNR = st.text_input('HNR')
            D2 = st.text_input('D2')
        with col2:
            fhi = st.text_input('MDVP:Fhi(Hz)')
            PPQ = st.text_input('MDVP:PPQ')
            APQ5 = st.text_input('Shimmer:APQ5')
            RPDE = st.text_input('RPDE')
            PPE = st.text_input('PPE')
        with col3:
            flo = st.text_input('MDVP:Flo(Hz)')
            DDP = st.text_input('Jitter:DDP')
            APQ = st.text_input('MDVP:APQ')
            DFA = st.text_input('DFA')
        with col4:
            Jitter_percent = st.text_input('MDVP:Jitter(%)')
            Shimmer = st.text_input('MDVP:Shimmer')
            DDA = st.text_input('Shimmer:DDA')
            spread1 = st.text_input('spread1')
        with col5:
            Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
            Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
            NHR = st.text_input('NHR')
            spread2 = st.text_input('spread2')

        submitted = st.form_submit_button("Get Parkinson's Test Result")

        if submitted:
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                          Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                          RPDE, DFA, spread1, spread2, D2, PPE]
            user_input = [handle_invalid_input(x) for x in user_input]

            if None in user_input:
                st.error('Please enter valid numerical inputs.')
            else:
                result = parkinsons_model.predict([user_input])[0]
                st.success("The person has Parkinson's disease." if result == 1 else "The person does not have Parkinson's disease.")
