import pandas as pd
import streamlit as st
import requests


def request_prediction(model_uri, columns, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'columns': columns, 'data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'

    st.title('Test dashboard octroi crédit')

    predict_btn = st.button('Prédire')
    if predict_btn:
        columns = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'CNT_FAM_MEMBERS',
       'DEF_30_CNT_SOCIAL_CIRCLE', 'CLIENT_AGE', 'ANNUAL_PAYMENT_RATE',
       'NAME_CONTRACT_TYPE_Cash loans',
       'NAME_CONTRACT_TYPE_Revolving loans', 'CODE_GENDER_F',
       'CODE_GENDER_M', 'FLAG_OWN_CAR_N', 'FLAG_OWN_CAR_Y',
       'FLAG_OWN_REALTY_N', 'FLAG_OWN_REALTY_Y',
       'NAME_INCOME_TYPE_Businessman',
       'NAME_INCOME_TYPE_Commercial associate',
       'NAME_INCOME_TYPE_Maternity leave', 'NAME_INCOME_TYPE_Pensioner',
       'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Student',
       'NAME_INCOME_TYPE_Unemployed', 'NAME_INCOME_TYPE_Working',
       'NAME_EDUCATION_TYPE_Academic degree',
       'NAME_EDUCATION_TYPE_Higher education',
       'NAME_EDUCATION_TYPE_Incomplete higher',
       'NAME_EDUCATION_TYPE_Lower secondary',
       'NAME_EDUCATION_TYPE_Secondary / secondary special',
       'NAME_FAMILY_STATUS_Civil marriage', 'NAME_FAMILY_STATUS_Married',
       'NAME_FAMILY_STATUS_Separated',
       'NAME_FAMILY_STATUS_Single / not married',
       'NAME_FAMILY_STATUS_Widow', 'NAME_HOUSING_TYPE_Co-op apartment',
       'NAME_HOUSING_TYPE_House / apartment',
       'NAME_HOUSING_TYPE_Municipal apartment',
       'NAME_HOUSING_TYPE_Office apartment',
       'NAME_HOUSING_TYPE_Rented apartment',
       'NAME_HOUSING_TYPE_With parents', 'OWN_CAR_TYPE_New car',
       'OWN_CAR_TYPE_No car', 'OWN_CAR_TYPE_Old car',
       'OWN_CAR_TYPE_Very old car', 'OWN_CAR_TYPE_Young car',
       'JOB_SENIORITY_Beginner', 'JOB_SENIORITY_Long seniority',
       'JOB_SENIORITY_Medium seniority', 'JOB_SENIORITY_New job',
       'JOB_SENIORITY_No job']
        data = [[0.2824738494741639, 5.266840222513807, 3.2303779908206507, 0.0, 4.8361273782736145, 
                1.430999071160864, 3.429882789669023, 0.0, 2.114145433640367, 0.0, 0.0, 2.1187488970422987,
                0.0, 2.165030500473145, 0.0, 0.0, 0.0, 2.587614428492914, 0.0, 0.0, 0.0, 0.0, 0.0, 2.3396133318681356,
                0.0, 0.0, 0.0, 3.35882536337421, 0.0, 0.0, 0.0, 0.0, 0.0, 3.162622794121506, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 3.0793552418580177, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5874142233155877]]

        pred = request_prediction(MLFLOW_URI, columns, data)[0]
        st.write('{}'.format(pred))


if __name__ == '__main__':
    main()
