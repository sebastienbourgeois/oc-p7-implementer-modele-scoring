import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Récupération du modèle pour sortir les prédictions
modele = pickle.load(open("modele.pkl", "rb"))

# Prédit si le client aura des problèmes de remboursement et le score échéant
@app.route('/predictions', methods=['POST'])
def predire_octroi_score_credit():
	# Récupération des données reçues via la requête
	query_parameters = request.get_json()
	std_donnees_client = query_parameters['std_donnees_client']
 
 	# Préparation de la réponse à envoyer
	predictions_client = {}
	predictions_client['problemes_remboursement'] = modele.predict(std_donnees_client).tolist()[0]
	predictions_client['score_remboursement_client'] = modele.predict_proba(std_donnees_client).tolist()[0][0]

	# Retourne la réponse au format json
	return jsonify(predictions_client)

if __name__ == "__main__":
	app.run(debug=True)