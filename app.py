from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# --- CONFIGURATION ---
api_key = os.environ.get("GEMINI_API_KEY")

# Configuration globale de la clé
if api_key:
    genai.configure(api_key=api_key)
else:
    print("ATTENTION: Clé API non trouvée dans les variables d'environnement")

@app.route('/')
def home():
    return "🚀 L'API KJM AI est en ligne !✅"

# --- ENDPOINT 1 : CHAT (Texte avec Gemini 1.5 Flash) ---
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    # Récupération du message
    user_message = request.args.get('message') or request.json.get('message')
    
    if not user_message:
        return jsonify({"error": "Message manquant"}), 400

    try:
        # On définit le modèle ici pour être sûr qu'il est chargé
        # Utilisation de 'gemini-1.5-flash' qui est rapide et gratuit
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content(user_message)
        
        return jsonify({
            "status": "success",
            "type": "text",
            "reponse": response.text
        })
    except Exception as e:
        # Si Flash échoue, l'erreur s'affichera ici
        return jsonify({
            "error": "Erreur lors de la génération du texte",
            "details": str(e)
        }), 500

# --- ENDPOINT 2 : IMAGE (Image avec Imagen 3) ---
@app.route('/image', methods=['GET', 'POST'])
def generate_image():
    prompt = request.args.get('prompt') or request.json.get('prompt')
    
    if not prompt:
        return jsonify({"error": "Description (prompt) manquante"}), 400

    try:
        # Utilisation du modèle Imagen 3
        imagen_model = genai.ImageGenerationModel("imagen-3.0-generate-001")
        
        results = imagen_model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1",
            safety_filter_level="block_only_high",
            person_generation="allow_adult"
        )

        # Traitement de l'image pour la renvoyer en Base64
        for image in results:
            img_byte_arr = BytesIO()
            image._pil_image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            base64_data = base64.b64encode(img_byte_arr).decode('utf-8')
            
            return jsonify({
                "status": "success",
                "type": "image_base64",
                "data": base64_data
            })

    except Exception as e:
        return jsonify({
            "error": "Erreur lors de la génération d'image (Vérifiez si votre clé supporte Imagen)",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
