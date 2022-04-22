import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go

def recuperer_infos_client(id_client):
	URI = 'http://127.0.0.1:5000/clients'
	parametres = {"id": id_client}

	reponse = requests.get(URI, params=parametres)

	if reponse.status_code != 200:
		raise Exception("Request failed with status {}, {}".format(reponse.status_code, reponse.text))

	return reponse.json() # renvoie un dictionnaire

def request_prediction(model_uri, columns, data):
	headers = {"Content-Type": "application/json"}

	data_json = {'columns': columns, 'data': data}
	response = requests.request(
			method='POST', headers=headers, url=model_uri, json=data_json)

	if response.status_code != 200:
			raise Exception(
					"Request failed with status {}, {}".format(response.status_code, response.text))

	return response.json()

def construire_jauge_score(score_remboursement_client):
	fig = go.Figure(
		go.Indicator(
		    mode = "gauge+number",
		    value = score_remboursement_client,
		    domain = {'x': [0, 1], 'y': [0, 1]},
		    title = {'text': "Probabilité de remboursement du crédit"},
		    gauge = {
		    	'axis': {'range': [None, 1]},
		    	'steps': [
		    		{'range': [0, 0.35], 'color': "tomato"},
		    		{'range': [0.65, 1], 'color': "lightgreen"}
		    	]
		    }
	    )
	)
	return fig

def main():
	with st.sidebar:
		st.title('Test dashboard octroi crédit')
		with st.form("envoi_id_client"):
			id_client = st.text_input('Entrez l\'ID du client :')
			formulaire_valide = st.form_submit_button("Valider")

	if formulaire_valide:
		infos_client = recuperer_infos_client(id_client)
		if infos_client:
			st.header('Client #{}'.format(id_client))
			#st.write('L\'ID client choisi est le : ', id_client)
			problemes_remboursement = infos_client['problemes_remboursement'][0]
			score_remboursement_client = infos_client['score_client'][0][0]
			if problemes_remboursement:
				st.info('Demande de crédit refusée')
			else:
				st.info('Demande de crédit acceptée')
			#st.write('Score client : ', infos_client['score_client'][0][0])
			jauge_score = construire_jauge_score(score_remboursement_client)
			st.plotly_chart(jauge_score)
			#if infos_client['problemes_remboursement']
			#df_infos_client = pd.DataFrame(data=infos_client, index=["valeur"])
			#st.dataframe(data=df_infos_client.T)
		else:
			st.error('L\'ID client choisi est soit invalide soit introuvable parmi les clients.')
	# st.write('L\'ID client choisi est le : ', id_client)
	# infos_client = recuperer_infos_client(id_client)
	# #dict_infos_client = json.loads(json_infos_client)
	# df_infos_client = pd.DataFrame(data=infos_client, index=["valeur"])
	# st.dataframe(data=df_infos_client.T)

	# st.write('Nombre de variable pour le client : ', df_infos_client.shape[1])

	# predict_btn = st.button('Prédire')
	# if predict_btn:
	# 	columns = [
	# OK	'AMT_INCOME_TOTAL', 
	# OK	'AMT_CREDIT', 
	# OK	'CNT_FAM_MEMBERS',
	# OK	'DEF_30_CNT_SOCIAL_CIRCLE', 
	# OK	'CLIENT_AGE', 
	# OK	'ANNUAL_PAYMENT_RATE',
	# OK	'NAME_CONTRACT_TYPE_Cash loans',
	# OK	'NAME_CONTRACT_TYPE_Revolving loans', 
	# OK	'CODE_GENDER_F',
	# OK	'CODE_GENDER_M', 
	# OK	'FLAG_OWN_CAR_N', 
	# OK	'FLAG_OWN_CAR_Y',
	# OK	'FLAG_OWN_REALTY_N', 
	# OK	'FLAG_OWN_REALTY_Y',
	# 	'NAME_INCOME_TYPE_Businessman',
	# OK	'NAME_INCOME_TYPE_Commercial associate',
	# OK	'NAME_INCOME_TYPE_Maternity leave', 
	# OK	'NAME_INCOME_TYPE_Pensioner',
	# OK	'NAME_INCOME_TYPE_State servant', 
	# OK	'NAME_INCOME_TYPE_Student',
	# OK	'NAME_INCOME_TYPE_Unemployed', 
	# OK	'NAME_INCOME_TYPE_Working',
	# OK	'NAME_EDUCATION_TYPE_Academic degree',
	# OK	'NAME_EDUCATION_TYPE_Higher education',
	# OK	'NAME_EDUCATION_TYPE_Incomplete higher',
	# OK	'NAME_EDUCATION_TYPE_Lower secondary',
	# OK	'NAME_EDUCATION_TYPE_Secondary / secondary special',
	# OK	'NAME_FAMILY_STATUS_Civil marriage', 
	# OK	'NAME_FAMILY_STATUS_Married',
	# OK	'NAME_FAMILY_STATUS_Separated',
	# OK	'NAME_FAMILY_STATUS_Single / not married',
	# OK	'NAME_FAMILY_STATUS_Widow', 
	# OK	'NAME_HOUSING_TYPE_Co-op apartment',
	# OK	'NAME_HOUSING_TYPE_House / apartment',
	# OK	'NAME_HOUSING_TYPE_Municipal apartment',
	# OK	'NAME_HOUSING_TYPE_Office apartment',
	# OK	'NAME_HOUSING_TYPE_Rented apartment',
	# OK	'NAME_HOUSING_TYPE_With parents', 
	# OK	'OWN_CAR_TYPE_New car',
	# OK	'OWN_CAR_TYPE_No car', 
	# OK	'OWN_CAR_TYPE_Old car',
	# OK	'OWN_CAR_TYPE_Very old car', 
	# OK	'OWN_CAR_TYPE_Young car',
	# OK	'JOB_SENIORITY_Beginner', 
	# OK	'JOB_SENIORITY_Long seniority',
	# OK	'JOB_SENIORITY_Medium seniority', 
	# OK	'JOB_SENIORITY_New job',
	# OK	'JOB_SENIORITY_No job'
	# 	]
	# 	data = [[0.2824738494741639, 5.266840222513807, 3.2303779908206507, 0.0, 4.8361273782736145, 
	# 	        1.430999071160864, 3.429882789669023, 0.0, 2.114145433640367, 0.0, 0.0, 2.1187488970422987,
	# 	        0.0, 2.165030500473145, 0.0, 0.0, 0.0, 2.587614428492914, 0.0, 0.0, 0.0, 0.0, 0.0, 2.3396133318681356,
	# 	        0.0, 0.0, 0.0, 3.35882536337421, 0.0, 0.0, 0.0, 0.0, 0.0, 3.162622794121506, 0.0, 0.0, 0.0, 0.0, 0.0,
	# 	        0.0, 3.0793552418580177, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5874142233155877]]

	# 	pred = request_prediction(MLFLOW_URI, columns, data)[0]
	# 	st.write('{}'.format(pred))

if __name__ == "__main__":
	main()