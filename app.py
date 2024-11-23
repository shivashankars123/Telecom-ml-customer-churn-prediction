# coding: utf-8

import pandas as pd
import pickle
import streamlit as st

# Load the model
model = pickle.load(open("model.sav", "rb"))
df_1 = pd.read_csv("first_telc.csv")

# Function to make prediction
def make_prediction(inputs):
    # Create DataFrame for new user inputpip install streamlit
    data = [inputs]
    new_df = pd.DataFrame(data, columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'
    ])
    
    # Concatenate with original dataframe and process
    df_2 = pd.concat([df_1, new_df], ignore_index=True)
    
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    
    # Drop column 'tenure' as it's not needed after binning
    df_2.drop(columns=['tenure'], axis=1, inplace=True)
    
    # One-hot encode categorical variables
    new_df_dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                           'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])

    # Make prediction
    single = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1]
    
    # Prepare the output message
    if single == 1:
        result_message = "This customer is likely to churn!!"
    else:
        result_message = "This customer is likely to continue!!"
    
    return result_message, probability[0] * 100

# Streamlit App
def app():
    st.title("Customer Churn Prediction")
    
    # Collect inputs from the user
    inputQuery1 = st.selectbox("Senior Citizen", ['0', '1'])
    inputQuery2 = st.number_input("Monthly Charges", min_value=0, max_value=1000, value=50)
    inputQuery3 = st.number_input("Total Charges", min_value=0, max_value=50000, value=1500)
    inputQuery4 = st.selectbox("Gender", ['Male', 'Female'])
    inputQuery5 = st.selectbox("Partner", ['Yes', 'No'])
    inputQuery6 = st.selectbox("Dependents", ['Yes', 'No'])
    inputQuery7 = st.selectbox("Phone Service", ['Yes', 'No'])
    inputQuery8 = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
    inputQuery9 = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    inputQuery10 = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    inputQuery11 = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    inputQuery12 = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
    inputQuery13 = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
    inputQuery14 = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
    inputQuery15 = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
    inputQuery16 = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    inputQuery17 = st.selectbox("Paperless Billing", ['Yes', 'No'])
    inputQuery18 = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
    inputQuery19 = st.number_input("Tenure (months)", min_value=1, max_value=72, value=12)

    # Button for prediction
    if st.button("Predict"):
        # Prepare inputs
        inputs = [
            inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7,
            inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
            inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19
        ]
        
        # Get prediction and confidence
        result_message, confidence = make_prediction(inputs)
        
        # Display the result
        st.write(f"Prediction: {result_message}")
        st.write(f"Confidence: {confidence:.2f}%")

# Run the Streamlit app
if __name__ == "__main__":
    app()
