import pickle
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

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

# Récupération des données à mettre à disposition
modele = pickle.load(open("modele.pkl", "rb"))
applications = pd.read_csv("data/brutes/application_test.csv", usecols=infos_descriptives)

# Récupération des infos descriptives d'un client
@app.route('/clients', methods=['GET'])
def get_client():

	query_parameters = request.args
	id_client = int(query_parameters['id'])

	client = applications[applications['SK_ID_CURR'] == id_client].copy()
	# nb_lignes = client.shape[0]
	# nb_cols = client.shape[1]
	# is_dataframe = isinstance(client, pd.DataFrame)
	infos_client = {}
	infos_client['id'] = str(client['SK_ID_CURR'][0])
	infos_client['type_contrat'] = str(client['NAME_CONTRACT_TYPE'][0])
	infos_client['genre'] = str(client['CODE_GENDER'][0])
	infos_client['possede_voiture'] = str(client['FLAG_OWN_CAR'][0])
	infos_client['possede_bien_immobilier'] = str(client['FLAG_OWN_REALTY'][0])
	infos_client['nb_enfants'] = str(client['CNT_CHILDREN'][0])
	infos_client['montant_revenus'] = str(client['AMT_INCOME_TOTAL'][0])
	infos_client['montant_credit'] = str(client['AMT_CREDIT'][0])
	infos_client['montant_annuite'] = str(client['AMT_ANNUITY'][0])
	infos_client['type_revenus'] = str(client['NAME_INCOME_TYPE'][0])
	infos_client['type_education'] = str(client['NAME_EDUCATION_TYPE'][0])
	infos_client['statut_familial'] = str(client['NAME_FAMILY_STATUS'][0])
	infos_client['type_logement'] = str(client['NAME_HOUSING_TYPE'][0])
	infos_client['age'] = str(client['DAYS_BIRTH'][0])
	infos_client['age_job_actuel'] = str(client['DAYS_EMPLOYED'][0])
	infos_client['age_voiture'] = str(client['OWN_CAR_AGE'][0])
	infos_client['type_activite_pro'] = str(client['OCCUPATION_TYPE'][0])
	infos_client['taille_foyer'] = str(client['CNT_FAM_MEMBERS'][0])
	return jsonify(infos_client)
	#return jsonify(test=is_dataframe, nb_lignes=nb_lignes, nb_cols=nb_cols, id_client=id_client)

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
	return jsonify(response)

@app.route('/', methods=['GET'])
def home():
	return "This is the home"

if __name__ == "__main__":
	app.run(debug=True)