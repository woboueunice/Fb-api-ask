from flask import Flask, request, jsonify
import google.generativeai as genai
import os

app = Flask(__name__)

# Configuration de l'IA (On récupère la clé depuis les variables d'environnement)
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Choix du modèle
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/')
def home():
    return "L'API KJM AI est en ligne !"

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    # Récupérer le message de l'utilisateur (supporte GET et POST)
    user_message = request.args.get('message') or request.json.get('message')
    
    if not user_message:
        return jsonify({"error": "Veuillez envoyer un message"}), 400

    try:
        # Demander à l'IA
        response = model.generate_content(user_message)
        return jsonify({
            "status": "success",
            "reponse": response.text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
    