import pickle
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Liste des infos descriptives des clients que l'on souhaite afficher dans le dashboard
infos_descriptives = [
	"SK_ID_CURR",
	"NAME_CONTRACT_TYPE",
	"CODE_GENDER",
	"FLAG_OWN_CAR",
	"FLAG_OWN_REALTY",
	"CNT_CHILDREN",
	"AMT_INCOME_TOTAL",
	"AMT_CREDIT",
	"AMT_ANNUITY",
	"NAME_INCOME_TYPE",
	"NAME_EDUCATION_TYPE",
	"NAME_FAMILY_STATUS",
	"NAME_HOUSING_TYPE",
	"DAYS_BIRTH",
	"DAYS_EMPLOYED",
	"OWN_CAR_AGE",
	"OCCUPATION_TYPE",
	"CNT_FAM_MEMBERS"
]
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

# Récupération des données à mettre à disposition
modele = pickle.load(open("modele.pkl", "rb"))
pipeline_pretraitements = pickle.load(open("pipeline_pretraitements.pkl", "rb"))
df_demandes_credit_brutes = pd.read_csv("data/brutes/application_test.csv", usecols=variables_a_conserver)

def calculer_age_client(df_demandes_credit):
	df_demandes_credit_avec_age_client = df_demandes_credit.copy()
	df_demandes_credit_avec_age_client["CLIENT_AGE"] = round(-df_demandes_credit_avec_age_client['DAYS_BIRTH']/365, 0)
	df_demandes_credit_avec_age_client.drop(columns=["DAYS_BIRTH"], inplace=True)
	return df_demandes_credit_avec_age_client

def calculer_duree_emploi(df_demandes_credit):
	df_demandes_credit_avec_duree_emploi = df_demandes_credit.copy()
	df_demandes_credit_avec_duree_emploi["EMPLOYMENT_DURATION"] = round(-df_demandes_credit_avec_duree_emploi['DAYS_EMPLOYED']/365, 0)
	df_demandes_credit_avec_duree_emploi.drop(columns=["DAYS_EMPLOYED"], inplace=True)
	return df_demandes_credit_avec_duree_emploi

def calculer_taux_remboursement_annuel(df_demandes_credit):
	df_demandes_credit_avec_taux_remboursement_annuel = df_demandes_credit.copy()
	df_demandes_credit_avec_taux_remboursement_annuel['ANNUAL_PAYMENT_RATE'] = df_demandes_credit_avec_taux_remboursement_annuel['AMT_ANNUITY']/df_demandes_credit_avec_taux_remboursement_annuel['AMT_CREDIT']
	df_demandes_credit_avec_taux_remboursement_annuel.drop(columns=["AMT_ANNUITY"], inplace=True)
	return df_demandes_credit_avec_taux_remboursement_annuel

def definir_anciennete_voiture(df_demandes_credit):
	df_demandes_credit_avec_anciennete_voiture = df_demandes_credit.copy()

	masque_pas_voiture = df_demandes_credit_avec_anciennete_voiture['OWN_CAR_AGE'].isnull()
	masque_voiture_neuve = df_demandes_credit_avec_anciennete_voiture['OWN_CAR_AGE'] <= 3
	masque_jeune_voiture = (df_demandes_credit_avec_anciennete_voiture['OWN_CAR_AGE'] >= 4) & (df_demandes_credit_avec_anciennete_voiture['OWN_CAR_AGE'] <= 9)
	masque_vieille_voiture = (df_demandes_credit_avec_anciennete_voiture['OWN_CAR_AGE'] >= 10) & (df_demandes_credit_avec_anciennete_voiture['OWN_CAR_AGE'] <= 19)
	masque_tres_vieille_voiture = df_demandes_credit_avec_anciennete_voiture['OWN_CAR_AGE'] >= 20

	df_demandes_credit_avec_anciennete_voiture.loc[masque_pas_voiture, 'OWN_CAR_TYPE'] = "No car"
	df_demandes_credit_avec_anciennete_voiture.loc[masque_voiture_neuve, 'OWN_CAR_TYPE'] = "New car"
	df_demandes_credit_avec_anciennete_voiture.loc[masque_jeune_voiture, 'OWN_CAR_TYPE'] = "Young car"
	df_demandes_credit_avec_anciennete_voiture.loc[masque_vieille_voiture, 'OWN_CAR_TYPE'] = "Old car"
	df_demandes_credit_avec_anciennete_voiture.loc[masque_tres_vieille_voiture, 'OWN_CAR_TYPE'] = "Very old car"

	df_demandes_credit_avec_anciennete_voiture.drop(columns=["OWN_CAR_AGE"], inplace=True)

	return df_demandes_credit_avec_anciennete_voiture

def definir_anciennete_emploi(df_demandes_credit):
	df_demandes_credit_avec_anciennete_emploi = df_demandes_credit.copy()

	masque_sans_activite = df_demandes_credit_avec_anciennete_emploi['EMPLOYMENT_DURATION'] == -1001
	masque_debutants = (df_demandes_credit_avec_anciennete_emploi['CLIENT_AGE'] <= 29) & (df_demandes_credit_avec_anciennete_emploi['EMPLOYMENT_DURATION'] >= 0) & (df_demandes_credit_avec_anciennete_emploi['EMPLOYMENT_DURATION'] <= 3)
	masque_nouveau_job = (df_demandes_credit_avec_anciennete_emploi['CLIENT_AGE'] >= 30) & (df_demandes_credit_avec_anciennete_emploi['EMPLOYMENT_DURATION'] >= 0) & (df_demandes_credit_avec_anciennete_emploi['EMPLOYMENT_DURATION'] <= 3)
	masque_confirmes = (df_demandes_credit_avec_anciennete_emploi['EMPLOYMENT_DURATION'] > 3) & (df_demandes_credit_avec_anciennete_emploi['EMPLOYMENT_DURATION'] <= 10)
	masque_anciens = df_demandes_credit_avec_anciennete_emploi['EMPLOYMENT_DURATION'] > 10

	df_demandes_credit_avec_anciennete_emploi.loc[masque_sans_activite, 'JOB_SENIORITY'] = "No job"
	df_demandes_credit_avec_anciennete_emploi.loc[masque_debutants, 'JOB_SENIORITY'] = "Beginner"
	df_demandes_credit_avec_anciennete_emploi.loc[masque_nouveau_job, 'JOB_SENIORITY'] = "New job"
	df_demandes_credit_avec_anciennete_emploi.loc[masque_confirmes, 'JOB_SENIORITY'] = "Medium seniority"
	df_demandes_credit_avec_anciennete_emploi.loc[masque_anciens, 'JOB_SENIORITY'] = "Long seniority"

	df_demandes_credit_avec_anciennete_emploi.drop(columns=["EMPLOYMENT_DURATION"], inplace=True)

	return df_demandes_credit_avec_anciennete_emploi

def generer_features_engineering(df_demandes_credit):
	df_demandes_credit_avec_nouvelles_variables = calculer_age_client(df_demandes_credit)
	df_demandes_credit_avec_nouvelles_variables = calculer_duree_emploi(df_demandes_credit_avec_nouvelles_variables)
	df_demandes_credit_avec_nouvelles_variables = definir_anciennete_voiture(df_demandes_credit_avec_nouvelles_variables)
	df_demandes_credit_avec_nouvelles_variables = definir_anciennete_emploi(df_demandes_credit_avec_nouvelles_variables)
	df_demandes_credit_avec_nouvelles_variables = calculer_taux_remboursement_annuel(df_demandes_credit_avec_nouvelles_variables)
	return df_demandes_credit_avec_nouvelles_variables

def renommer_colonnes(df_demandes_credit, pipeline_pretraitements):
	nom_colonnes_pipeline = pipeline_pretraitements.get_feature_names_out(df_demandes_credit.columns)
	nom_colonnes = []
	for colonne in nom_colonnes_pipeline:
	    if colonne[0:13] == "categorielles":
	        nom_colonnes.append(colonne[15:])
	    else:
	        nom_colonnes.append(colonne[11:])
	return nom_colonnes

def standardiser_data(df_demandes_credit, pipeline_pretraitements):
	df_demandes_credit_sans_ID = df_demandes_credit.drop(columns=['SK_ID_CURR'])
	nom_colonnes = renommer_colonnes(df_demandes_credit_sans_ID, pipeline_pretraitements)
	df_std_demandes_credit = pd.DataFrame(data=pipeline_pretraitements.transform(df_demandes_credit_sans_ID), 
										  columns=nom_colonnes)
	return df_std_demandes_credit

# df_demandes_credit = generer_features_engineering(df_demandes_credit_brutes)
# df_std_demandes_credit = standardiser_data(df_demandes_credit, pipeline_pretraitements)
# df_id_client = df_demandes_credit[['SK_ID_CURR']].copy()
# df_std_demandes_credit_avec_id_client = df_id_client.merge(df_std_demandes_credit,
# 														how = 'inner',
# 														left_index=True,
# 														right_index=True,
# 														suffixes=(False, False))

# Récupération des infos descriptives d'un client
@app.route('/client', methods=['GET'])
def get_client_brut():

	query_parameters = request.args
	id_client = int(query_parameters['id'])

	client = applications_OK[applications_OK['SK_ID_CURR'] == id_client].copy()
	# nb_lignes = client.shape[0]
	# nb_cols = client.shape[1]
	# is_dataframe = isinstance(client, pd.DataFrame)
	infos_client = {}
	infos_client['id'] = str(client['SK_ID_CURR'][0])
	infos_client['type_contrat'] = str(client['NAME_CONTRACT_TYPE'][0])
	infos_client['genre'] = str(client['CODE_GENDER'][0])
	infos_client['possede_voiture'] = str(client['FLAG_OWN_CAR'][0])
	infos_client['possede_bien_immobilier'] = str(client['FLAG_OWN_REALTY'][0])
	infos_client['montant_revenus'] = str(client['AMT_INCOME_TOTAL'][0])
	infos_client['montant_credit'] = str(client['AMT_CREDIT'][0])
	infos_client['type_revenus'] = str(client['NAME_INCOME_TYPE'][0])
	infos_client['type_education'] = str(client['NAME_EDUCATION_TYPE'][0])
	infos_client['statut_familial'] = str(client['NAME_FAMILY_STATUS'][0])
	infos_client['type_logement'] = str(client['NAME_HOUSING_TYPE'][0])
	infos_client['taille_foyer'] = str(client['CNT_FAM_MEMBERS'][0])
	infos_client['cercle_social_en_defaut_30j'] = str(client['DEF_30_CNT_SOCIAL_CIRCLE'][0])
	infos_client['age'] = str(client['CLIENT_AGE'][0])
	infos_client['type_voiture'] = str(client['OWN_CAR_TYPE'][0])
	infos_client['anciennete_job'] = str(client['JOB_SENIORITY'][0])
	infos_client['taux_remboursement_annuel'] = str(client['ANNUAL_PAYMENT_RATE'][0])
	return jsonify(infos_client)
	#return jsonify(test=is_dataframe, nb_lignes=nb_lignes, nb_cols=nb_cols, id_client=id_client)

infos_descriptives = [
	"SK_ID_CURR",
	"NAME_CONTRACT_TYPE",
	"CODE_GENDER",
	"FLAG_OWN_CAR",
	"FLAG_OWN_REALTY",
	"CNT_CHILDREN",
	"AMT_INCOME_TOTAL",
	"AMT_CREDIT",
	"AMT_ANNUITY",
	"NAME_INCOME_TYPE",
	"NAME_EDUCATION_TYPE",
	"NAME_FAMILY_STATUS",
	"NAME_HOUSING_TYPE",
	"DAYS_BIRTH",
	"DAYS_EMPLOYED",
	"OWN_CAR_AGE",
	"OCCUPATION_TYPE",
	"CNT_FAM_MEMBERS"
]

def id_client_est_trouve(id_client):
	nb_client_trouve = len(df_demandes_credit_brutes[df_demandes_credit_brutes['SK_ID_CURR'] == id_client])
	if nb_client_trouve >= 1:
		return True
	else:
		return False

def id_client_recu_est_verifie(id_client_recu):
	if id_client_recu and id_client_est_trouve(int(id_client_recu)):
		return True
	else:
		return False

@app.route('/clients', methods=['GET'])
def recuperer_infos_demande_credit_client():
	query_parameters = request.args
	infos_client = {}
	if id_client_recu_est_verifie(query_parameters['id']):
		id_client = int(query_parameters['id'])

		df_data_brutes_client = df_demandes_credit_brutes[df_demandes_credit_brutes['SK_ID_CURR'] == id_client].copy()
		df_data_client = generer_features_engineering(df_data_brutes_client)
		df_std_data_client = standardiser_data(df_data_client, pipeline_pretraitements)

		infos_client['problemes_remboursement'] = modele.predict(df_std_data_client).tolist()
		infos_client['score_client'] = modele.predict_proba(df_std_data_client).tolist()

		infos_client['id'] = str(df_data_client['SK_ID_CURR'][0])
		infos_client['type_contrat'] = str(df_data_client['NAME_CONTRACT_TYPE'][0])
		infos_client['genre'] = str(df_data_client['CODE_GENDER'][0])
		infos_client['possede_voiture'] = str(df_data_client['FLAG_OWN_CAR'][0])
		infos_client['possede_bien_immobilier'] = str(df_data_client['FLAG_OWN_REALTY'][0])
		infos_client['montant_revenus'] = str(df_data_client['AMT_INCOME_TOTAL'][0])
		infos_client['montant_credit'] = str(df_data_client['AMT_CREDIT'][0])
		infos_client['type_revenus'] = str(df_data_client['NAME_INCOME_TYPE'][0])
		infos_client['type_education'] = str(df_data_client['NAME_EDUCATION_TYPE'][0])
		infos_client['statut_familial'] = str(df_data_client['NAME_FAMILY_STATUS'][0])
		infos_client['type_logement'] = str(df_data_client['NAME_HOUSING_TYPE'][0])
		infos_client['taille_foyer'] = str(df_data_client['CNT_FAM_MEMBERS'][0])
		infos_client['cercle_social_en_defaut_30j'] = str(df_data_client['DEF_30_CNT_SOCIAL_CIRCLE'][0])
		infos_client['age'] = str(df_data_client['CLIENT_AGE'][0])
		infos_client['type_voiture'] = str(df_data_client['OWN_CAR_TYPE'][0])
		infos_client['anciennete_job'] = str(df_data_client['JOB_SENIORITY'][0])
		infos_client['taux_remboursement_annuel'] = str(df_data_client['ANNUAL_PAYMENT_RATE'][0])
	
	return jsonify(infos_client)

# Récupération des infos descriptives d'un client
#@app.route('/clients', methods=['GET'])
def get_client():

	query_parameters = request.args
	id_client = int(query_parameters['id'])

	client = applications[applications['SK_ID_CURR'] == id_client].copy()

	client_OK = generer_features_engineering(client)
	std_client = standardiser_data(client_OK, pipeline_pretraitements)
	# nb_lignes = client.shape[0]
	# nb_cols = client.shape[1]
	# is_dataframe = isinstance(client, pd.DataFrame)
	infos_client = {}
	infos_client['id'] = str(client['SK_ID_CURR'][0])
	infos_client['type_contrat'] = str(client_OK['NAME_CONTRACT_TYPE'][0])
	infos_client['genre'] = str(client_OK['CODE_GENDER'][0])
	infos_client['possede_voiture'] = str(client_OK['FLAG_OWN_CAR'][0])
	infos_client['possede_bien_immobilier'] = str(client_OK['FLAG_OWN_REALTY'][0])
	infos_client['montant_revenus'] = str(client_OK['AMT_INCOME_TOTAL'][0])
	infos_client['montant_credit'] = str(client_OK['AMT_CREDIT'][0])
	infos_client['type_revenus'] = str(client_OK['NAME_INCOME_TYPE'][0])
	infos_client['type_education'] = str(client_OK['NAME_EDUCATION_TYPE'][0])
	infos_client['statut_familial'] = str(client_OK['NAME_FAMILY_STATUS'][0])
	infos_client['type_logement'] = str(client_OK['NAME_HOUSING_TYPE'][0])
	infos_client['taille_foyer'] = str(client_OK['CNT_FAM_MEMBERS'][0])
	infos_client['cercle_social_en_defaut_30j'] = str(client_OK['DEF_30_CNT_SOCIAL_CIRCLE'][0])
	infos_client['age'] = str(client_OK['CLIENT_AGE'][0])
	infos_client['type_voiture'] = str(client_OK['OWN_CAR_TYPE'][0])
	infos_client['anciennete_job'] = str(client_OK['JOB_SENIORITY'][0])
	infos_client['taux_remboursement_annuel'] = str(client_OK['ANNUAL_PAYMENT_RATE'][0])

	# Old
	# infos_client = {}
	# infos_client['id'] = str(client['SK_ID_CURR'][0])
	# infos_client['type_contrat'] = str(client['NAME_CONTRACT_TYPE'][0])
	# infos_client['genre'] = str(client['CODE_GENDER'][0])
	# infos_client['possede_voiture'] = str(client['FLAG_OWN_CAR'][0])
	# infos_client['possede_bien_immobilier'] = str(client['FLAG_OWN_REALTY'][0])
	# infos_client['montant_revenus'] = str(client['AMT_INCOME_TOTAL'][0])
	# infos_client['montant_credit'] = str(client['AMT_CREDIT'][0])
	# infos_client['type_revenus'] = str(client['NAME_INCOME_TYPE'][0])
	# infos_client['type_education'] = str(client['NAME_EDUCATION_TYPE'][0])
	# infos_client['statut_familial'] = str(client['NAME_FAMILY_STATUS'][0])
	# infos_client['type_logement'] = str(client['NAME_HOUSING_TYPE'][0])
	# infos_client['taille_foyer'] = str(client['CNT_FAM_MEMBERS'][0])
	# infos_client['cercle_social_en_defaut_30j'] = str(client['DEF_30_CNT_SOCIAL_CIRCLE'][0])
	# infos_client['age'] = str(client['DAYS_BIRTH'][0])
	# infos_client['age_voiture'] = str(client['OWN_CAR_AGE'][0])
	# infos_client['anciennete_job'] = str(client['DAYS_EMPLOYED'][0])
	
	return jsonify(infos_client)

@app.route('/clients', methods=['GET'])
def predire_octroi_credit():
	query_parameters = request.args
	id_client = int(query_parameters['id'])

	client = applications[applications['SK_ID_CURR'] == id_client].copy()

	client_OK = generer_features_engineering(client)
	std_client = standardiser_data(client_OK, pipeline_pretraitements)

	predictions = {}
	#predictions['credit_accepte'] = modele.predict(std_client).tolist()

	return jsonify(list(std_client.columns))

# Defining a route for only post requests
@app.route('/predict', methods=['POST'])
def predict():
	# getting an array of features from the post request's body
	query_parameters = request.args
	feature_array = np.fromstring(query_parameters['feature_array'], dtype=float, sep=",")

	# creating a response object
	# storing the model's prediction in the object
	response = {}
	#response['type_prediction'] = type(modele.predict([feature_array]))
	response['predictions'] = modele.predict([feature_array]).tolist()

	# returning the response object as json
	return jsonify(len(feature_array))

@app.route('/', methods=['GET'])
def home():
	return "This is the home"

if __name__ == "__main__":
	app.run(debug=True)