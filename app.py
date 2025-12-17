from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import requests
import json

app = Flask(__name__)

# --- CONFIGURATION ---
api_key = os.environ.get("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)

@app.route('/')
def home():
    return "🚀 L'API KJM AI (Mode Gemmini 3 Pro) est en ligne !✅"

# --- ENDPOINT 1 : CHAT (Texte) ---
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    user_message = request.args.get('message') or request.json.get('message')
    
    if not user_message: return jsonify({"error": "Message manquant"}), 400

    try:
        # CHANGEMENT MAJEUR ICI 🏆
        # On utilise Gemma 3 12B IT car il offre 14 400 requêtes/jour
        # au lieu de 20 pour Gemini 2.5
        model = genai.GenerativeModel('gemma-3-12b-it')
        
        response = model.generate_content(user_message)
        
        return jsonify({
            "status": "success",
            "type": "text",
            "model_used": "gemma-3-12b-it",
            "reponse": response.text
        })
    except Exception as e:
        return jsonify({
            "error": "Erreur de génération", 
            "details": str(e),
            "conseil": "Vérifiez que votre clé API a accès aux modèles Gemma sur AI Studio"
        }), 500

# --- ENDPOINT 2 : IMAGE (Reste sur Imagen 3) ---
@app.route('/image', methods=['GET', 'POST'])
def generate_image():
    prompt = request.args.get('prompt') or request.json.get('prompt')
    if not prompt: return jsonify({"error": "Prompt manquant"}), 400
    
    # Pour l'image, on reste sur la méthode HTTP directe vers Imagen
    # car Gemma ne fait que du texte.
    try:
        if not api_key: return jsonify({"error": "Clé API manquante"}), 500
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predict?key={api_key}"
        
        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {"sampleCount": 1, "aspectRatio": "1:1", "personGeneration": "allow_adult"}
        }
        
        response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        
        if response.status_code != 200:
            return jsonify({"error": "Erreur Google Image", "msg": response.text})

        result = response.json()
        
        if 'predictions' in result and result['predictions']:
            base64_data = result['predictions'][0]['bytesBase64Encoded']
            return jsonify({"status": "success", "type": "image_base64", "data": base64_data})
        else:
            return jsonify({"error": "Pas d'image générée", "debug": result})

    except Exception as e:
        return jsonify({"error": "Erreur système", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
                
