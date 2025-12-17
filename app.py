from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import requests
import json
import base64

app = Flask(__name__)

# --- CONFIGURATION ---
api_key = os.environ.get("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)

@app.route('/')
def home():
    return "🚀 L'API KJM AI ( Version 2.5 ) est en ligne !✅"

# --- ENDPOINT 1 : CHAT (Texte - Multi-modèles) ---
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    user_message = request.args.get('message') or request.json.get('message')
    
    if not user_message: return jsonify({"error": "Message manquant"}), 400

    # LISTE DES MODÈLES À TESTER (Du meilleur au secours)
    # 1. Le Lite (Rapide, gratuit, gros quota)
    # 2. Le Flash Exp (Puissant, bon quota)
    models_to_try = [
        'gemini-2.0-flash-lite-preview-02-05',
        'gemini-2.0-flash-exp'
    ]

    last_error = ""

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(user_message)
            
            return jsonify({
                "status": "success",
                "type": "text",
                "model_used": model_name,
                "reponse": response.text
            })
        except Exception as e:
            # Si un modèle échoue (quota ou erreur), on passe au suivant dans la boucle
            last_error = str(e)
            continue
    
    # Si on arrive ici, c'est que TOUS les modèles ont échoué
    return jsonify({
        "error": "Tous les modèles sont occupés ou hors quota.", 
        "details": last_error
    }), 429

# --- ENDPOINT 2 : IMAGE (Connexion Directe HTTP) ---
@app.route('/image', methods=['GET', 'POST'])
def generate_image():
    prompt = request.args.get('prompt') or request.json.get('prompt')
    if not prompt: return jsonify({"error": "Prompt manquant"}), 400
    if not api_key: return jsonify({"error": "Clé API manquante"}), 500

    try:
        # On appelle directement l'API Web de Google (contourne le bug de la librairie Render)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predict?key={api_key}"
        
        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": "1:1",
                "personGeneration": "allow_adult"
            }
        }
        
        # Envoi de la requête
        response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        
        if response.status_code != 200:
            return jsonify({"error": "Erreur Google Image", "code": response.status_code, "msg": response.text})

        result = response.json()
        
        # Extraction de l'image
        if 'predictions' in result and result['predictions']:
            base64_data = result['predictions'][0]['bytesBase64Encoded']
            return jsonify({
                "status": "success",
                "type": "image_base64",
                "data": base64_data
            })
        else:
            return jsonify({"error": "Pas d'image générée", "debug": result})

    except Exception as e:
        return jsonify({"error": "Erreur système", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
