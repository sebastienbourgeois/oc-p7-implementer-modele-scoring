import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@st.cache
def charger_demandes_credit(chemin_fichier_donnees):
	variables_a_conserver = [
		'SK_ID_CURR',
		'NAME_CONTRACT_TYPE',
		'CODE_GENDER',
		'FLAG_OWN_CAR',
		'FLAG_OWN_REALTY',
		'AMT_INCOME_TOTAL',
		'AMT_CREDIT',
		'NAME_INCOME_TYPE',
		'NAME_EDUCATION_TYPE',
		'NAME_FAMILY_STATUS',
		'NAME_HOUSING_TYPE',
		'CNT_FAM_MEMBERS',
		'DEF_30_CNT_SOCIAL_CIRCLE',
		'DAYS_BIRTH',
		'OWN_CAR_AGE',
		'DAYS_EMPLOYED',
		'AMT_ANNUITY'
	]
	return pd.read_csv(chemin_fichier_donnees, usecols=variables_a_conserver)

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

def recuperer_liste_id_clients(df_demandes_credit):
	return df_demandes_credit['SK_ID_CURR'].tolist()

def calculer_age_client(df_demandes_credit):
	df_demandes_credit["CLIENT_AGE"] = round(-df_demandes_credit['DAYS_BIRTH']/365, 0)
	df_demandes_credit.drop(columns=["DAYS_BIRTH"], inplace=True)
	return df_demandes_credit

def calculer_duree_emploi(df_demandes_credit):
	df_demandes_credit["EMPLOYMENT_DURATION"] = round(-df_demandes_credit['DAYS_EMPLOYED']/365, 0)
	df_demandes_credit.drop(columns=["DAYS_EMPLOYED"], inplace=True)
	return df_demandes_credit

def calculer_taux_remboursement_annuel(df_demandes_credit):
	df_demandes_credit['ANNUAL_PAYMENT_RATE'] = df_demandes_credit['AMT_ANNUITY']/df_demandes_credit['AMT_CREDIT']
	df_demandes_credit.drop(columns=["AMT_ANNUITY"], inplace=True)
	return df_demandes_credit

def definir_anciennete_voiture(df_demandes_credit):
	masque_pas_voiture = df_demandes_credit['OWN_CAR_AGE'].isnull()
	masque_voiture_neuve = df_demandes_credit['OWN_CAR_AGE'] <= 3
	masque_jeune_voiture = (df_demandes_credit['OWN_CAR_AGE'] >= 4) & (df_demandes_credit['OWN_CAR_AGE'] <= 9)
	masque_vieille_voiture = (df_demandes_credit['OWN_CAR_AGE'] >= 10) & (df_demandes_credit['OWN_CAR_AGE'] <= 19)
	masque_tres_vieille_voiture = df_demandes_credit['OWN_CAR_AGE'] >= 20

	df_demandes_credit.loc[masque_pas_voiture, 'OWN_CAR_TYPE'] = "No car"
	df_demandes_credit.loc[masque_voiture_neuve, 'OWN_CAR_TYPE'] = "New car"
	df_demandes_credit.loc[masque_jeune_voiture, 'OWN_CAR_TYPE'] = "Young car"
	df_demandes_credit.loc[masque_vieille_voiture, 'OWN_CAR_TYPE'] = "Old car"
	df_demandes_credit.loc[masque_tres_vieille_voiture, 'OWN_CAR_TYPE'] = "Very old car"

	df_demandes_credit.drop(columns=["OWN_CAR_AGE"], inplace=True)

	return df_demandes_credit

def definir_anciennete_emploi(df_demandes_credit):
	masque_sans_activite = df_demandes_credit['EMPLOYMENT_DURATION'] == -1001
	masque_debutants = (df_demandes_credit['CLIENT_AGE'] <= 29) & (df_demandes_credit['EMPLOYMENT_DURATION'] >= 0) & (df_demandes_credit['EMPLOYMENT_DURATION'] <= 3)
	masque_nouveau_job = (df_demandes_credit['CLIENT_AGE'] >= 30) & (df_demandes_credit['EMPLOYMENT_DURATION'] >= 0) & (df_demandes_credit['EMPLOYMENT_DURATION'] <= 3)
	masque_confirmes = (df_demandes_credit['EMPLOYMENT_DURATION'] > 3) & (df_demandes_credit['EMPLOYMENT_DURATION'] <= 10)
	masque_anciens = df_demandes_credit['EMPLOYMENT_DURATION'] > 10

	df_demandes_credit.loc[masque_sans_activite, 'JOB_SENIORITY'] = "No job"
	df_demandes_credit.loc[masque_debutants, 'JOB_SENIORITY'] = "Beginner"
	df_demandes_credit.loc[masque_nouveau_job, 'JOB_SENIORITY'] = "New job"
	df_demandes_credit.loc[masque_confirmes, 'JOB_SENIORITY'] = "Medium seniority"
	df_demandes_credit.loc[masque_anciens, 'JOB_SENIORITY'] = "Long seniority"

	df_demandes_credit.drop(columns=["EMPLOYMENT_DURATION"], inplace=True)

	return df_demandes_credit

@st.cache
def generer_features_engineering(df_demandes_credit_brutes):
	df_demandes_credit = df_demandes_credit_brutes.copy()
	df_demandes_credit = calculer_age_client(df_demandes_credit)
	df_demandes_credit = calculer_duree_emploi(df_demandes_credit)
	df_demandes_credit = definir_anciennete_voiture(df_demandes_credit)
	df_demandes_credit = definir_anciennete_emploi(df_demandes_credit)
	df_demandes_credit = calculer_taux_remboursement_annuel(df_demandes_credit)
	return df_demandes_credit

def creer_pipeline_pretraitements(df_demandes_credit):
	s = (df_demandes_credit.dtypes == 'object')
	variables_categorielles = list(s[s].index)

	categorical_transformer = ColumnTransformer(
	    transformers=[
	        ('categorielles', OneHotEncoder(handle_unknown='ignore', sparse=False), variables_categorielles)
	    ],
	    remainder = 'passthrough'
	)

	preprocessor = Pipeline(steps=[
	    ('encodage', categorical_transformer),
	    ('standardisation', StandardScaler(with_mean=False))
	])

	return preprocessor.fit(df_demandes_credit)

def renommer_colonnes(df_demandes_credit, pipeline_pretraitements):
	nom_colonnes_pipeline = pipeline_pretraitements.get_feature_names_out(df_demandes_credit.columns)
	nom_colonnes = []
	for colonne in nom_colonnes_pipeline:
	    if colonne[0:13] == "categorielles":
	        nom_colonnes.append(colonne[15:])
	    else:
	        nom_colonnes.append(colonne[11:])
	return nom_colonnes

@st.cache
def standardiser_data(df_demandes_credit):
	df_demandes_credit_sans_ID = df_demandes_credit.drop(columns=['SK_ID_CURR'])
	pipeline_pretraitements = creer_pipeline_pretraitements(df_demandes_credit_sans_ID)
	nom_colonnes = renommer_colonnes(df_demandes_credit_sans_ID, pipeline_pretraitements)
	df_std_demandes_credit = pd.DataFrame(data=pipeline_pretraitements.transform(df_demandes_credit_sans_ID), 
										  columns=nom_colonnes)
	return df_std_demandes_credit

def recuperer_liste_variables(df_demandes_credit):
	liste_variables = df_demandes_credit.columns.tolist()
	liste_variables.remove('SK_ID_CURR')
	return liste_variables

#def mettre_a_jour_autre_liste_variables():
#	st.state_session.

def construire_graphique(df_demandes_credit, variable):
	#type_variable = df_demandes_credit[variable].dtype.name
	#if type_variable == 'object':
	fig = go.Figure(data=[go.Histogram(x=df_demandes_credit[variable])])
	return fig

def recuperer_donnee_client(df_demandes_credit, id_client, variable):
	df_tmp = df_demandes_credit[df_demandes_credit['SK_ID_CURR'] == id_client]
	df_tmp = df_tmp.reset_index(drop=True)
	donnee_client = df_tmp.loc[0, variable]
	return donnee_client

def ajouter_position_client(graphique, valeur_client):
	graphique.add_annotation(x=valeur_client, y=0, text="<b>Position du client</b>", 
		showarrow=True, arrowhead=1, arrowwidth=2,
		bordercolor="black", borderwidth=1,
		bgcolor="white")
	return graphique

def main():

	df_demandes_credit_brutes = charger_demandes_credit("data/brutes/application_test.csv")
	#data_preprocessing_state = st.text('Preprocessing data...')
	df_demandes_credit = generer_features_engineering(df_demandes_credit_brutes)
	df_std_demandes_credit = standardiser_data(df_demandes_credit)
	#data_preprocessing_state.text('Preprocessing data...done! (w/ cache)')
	#data_preprocessing_state.text('Preprocessing data...done!')

	with st.sidebar:
		st.title('Test dashboard octroi crédit')
		liste_id_clients = recuperer_liste_id_clients(df_demandes_credit)
		#st.write(liste_id_clients['id'])
		#with st.form("envoi_id_client"):

			#id_client = st.text_input('Entrez l\'ID du client :')
			#id_client = st.selectbox('Entrez l\'ID du client :', (100001, 100005))
		id_client = st.selectbox('ID client', liste_id_clients)
			#formulaire_valide = st.form_submit_button("Valider")
		liste_variables = recuperer_liste_variables(df_demandes_credit)
		variable1 = st.selectbox('Feature 1 à afficher', liste_variables, index=0, key="feature1")
		variable2 = st.selectbox('Feature 2 à afficher', liste_variables, index=1, key="feature2")

	#if formulaire_valide:
	infos_client = recuperer_infos_client(id_client)
	#st.dataframe(df_demandes_credit[df_demandes_credit['SK_ID_CURR'] == id_client])
	if infos_client:
		st.header('Client #{}'.format(id_client))
		#st.write('L\'ID client choisi est le : ', id_client)
		#st.write(infos_client['problemes_remboursement'])
		#st.write(infos_client['score_client'])
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

		colonne_gauche, colonne_droite = st.columns(2)

		#with colonne_gauche:
		graphique1 = construire_graphique(df_demandes_credit, variable1)
		donnee1_client = recuperer_donnee_client(df_demandes_credit, id_client, variable1)
		graphique1 = ajouter_position_client(graphique1, donnee1_client)		
		st.plotly_chart(graphique1)

		#with colonne_droite:
		graphique2 = construire_graphique(df_demandes_credit, variable2)
		donnee2_client = recuperer_donnee_client(df_demandes_credit, id_client, variable2)
		graphique2 = ajouter_position_client(graphique2, donnee2_client)	
		st.plotly_chart(graphique2)

		#else:
		#	st.error('L\'ID client choisi est soit invalide soit introuvable parmi les clients.')
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