# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install seaborn
pip install scikit-learn
# pip install streamlit


import pickle
import pandas as pd
import numpy as np
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('gbc_model.pkl', 'rb'))

# Creating a function for prediction

def census_income_prediction(input_data):

    input = np.asarray(input_data)
    input_reshape = input.reshape(1, -1)

    prediction = loaded_model.predict(input_reshape)

    print(prediction)

    if prediction[0] == 0:
        return "Annual income is less than 50K"
    else:
        return "Annual income is  more than 50K"
    
workclass_cat = {0: 'Federal-gov', 1: 'Local-gov', 2: 'Never-worked', 3: 'Private', 
                      4: 'Self-emp-inc', 5: 'Self-emp-not-inc', 6: 'State-gov', 7: 'Without-pay'}

education_cat = {1: 'Preschool', 2: '1st-4th', 3: '5th-6th', 4: '7th-8th', 5: '9th', 6: '10th', 7: '11th', 
                 8: '12th', 9: 'HS-grad', 10: 'Some-college', 11: 'Assoc-voc', 12: 'Assoc-acdm', 13: 'Bachelors', 
                 14: 'Masters', 15: 'Prof-school', 16: 'Doctorate'}

marital_status_cat = {0: 'Divorced', 1: 'Married-AF-spouse', 2: 'Married-civ-spouse', 3: 'Married-spouse-absent', 
                  4: 'Never-married', 5: 'Separated', 6: 'Widowed'}

occupation_cat = {0: 'Adm-clerical', 1: 'Armed-Forces', 2: 'Craft-repair', 3: 'Exec-managerial', 4: 'Farming-fishing', 5: 'Handlers-cleaners', 
              6: 'Machine-op-inspct', 7: 'Other-service', 8: 'Priv-house-serv', 9: 'Prof-specialty', 10: 'Protective-serv', 11: 'Sales', 
              12: 'Tech-support', 13: 'Transport-moving'}

relationship_cat = {0: 'Husband', 1: 'Not-in-family', 2: 'Other-relative', 3: 'Own-child', 4: 'Unmarried', 5: 'Wife'}

race_cat = {0: 'Amer-Indian-Eskimo', 1: 'Asian-Pac-Islander', 2: 'Black', 3: 'Other', 4: 'White'}

gender_cat = {0: 'Female', 1: 'Male'}

native_country_cat = {0: 'Cambodia', 1: 'Canada', 2: 'China', 3: 'Columbia', 4: 'Cuba', 5: 'Dominican-Republic', 6: 'Ecuador', 
                  7: 'El-Salvador', 8: 'England', 9: 'France', 10: 'Germany', 11: 'Greece', 12: 'Guatemala', 13: 'Haiti', 
                  14: 'Holand-Netherlands', 15: 'Honduras', 16: 'Hong', 17: 'Hungary', 18: 'India', 19: 'Iran', 20: 'Ireland', 
                  21: 'Italy', 22: 'Jamaica', 23: 'Japan', 24: 'Laos', 25: 'Mexico', 26: 'Nicaragua', 27: 'Outlying-US(Guam-USVI-etc)', 
                  28: 'Peru', 29: 'Philippines', 30: 'Poland', 31: 'Portugal', 32: 'Puerto-Rico', 33: 'Scotland', 34: 'South', 
                  35: 'Taiwan', 36: 'Thailand', 37: 'Trinadad&Tobago', 38: 'United-States', 39: 'Vietnam', 40: 'Yugoslavia'}

    

def main():

    # Giving a title
    st.title("Census Income Prediction Web App")

    # Getting the input data from the user
    age = st.text_input("Enter your age")

    workclass = st.selectbox("Select Work Class", options=list(workclass_cat.keys()),
                             format_func=lambda x: workclass_cat[x])
    st.write(f"Selected Work Class: {workclass}")

    final_weight = st.text_input('Enter Final weight')

    education = st.selectbox("Select Education", options = list(education_cat.keys()),
                            format_func = lambda x: education_cat[x])
    st.write(f"Selected Education: {education}")

    marital_status = st.selectbox('Select Marital Status', options= list(marital_status_cat.keys()),
                                  format_func = lambda x: marital_status_cat[x])
    st.write(f"Selected Marital Status: {marital_status}")

    occupation = st.selectbox("Select Occupation", options=list(occupation_cat.keys()),
                              format_func=lambda x: occupation_cat[x])
    st.write(f"Selected Occupation: {occupation}")

    relationship = st.selectbox("Select Relationship", options=list(relationship_cat.keys()),
                                format_func=lambda x: relationship_cat[x])
    st.write(f"Selected Relationship {relationship}")

    race = st.selectbox("Select Race", options = list(race_cat.keys()),
                        format_func = lambda x: race_cat[x])
    st.write(f"Selected Race {race}")

    gender = st.selectbox("Select Gender", options = list(gender_cat.keys()),
                          format_func=lambda x: gender_cat[x])
    st.write(f"Selected Gender {gender}")

    captail_gain = st.text_input("Enter Captial Gain")

    captail_loss = st.text_input("Enter Captial loss")

    hrs_per_week = st.text_input("Enter Hours per week")

    native_country = st.selectbox("Enter Native Country", options=list(native_country_cat.keys()),
                                  format_func=lambda x: native_country_cat[x])
    st.write(f"Selected Native Country {native_country}")


    # Code for prediction
    predicted_income = ''

    # Creating a button for prediction
    if st.button('Income prediction'):
        predicted_income = census_income_prediction([age, workclass, final_weight, education, marital_status, occupation, relationship,
                                                     race, gender, captail_gain, captail_loss, hrs_per_week, native_country ])
        
    st.success(predicted_income)

if __name__ == "__main__":
    main()