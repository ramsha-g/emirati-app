# 🟢 Flask + Whisper + ngrok Transcription Server

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import whisper
import tempfile
import os
from flask_cors import CORS
from pyngrok import ngrok

# Load Whisper model (use "base", "small", "medium", etc)
model = whisper.load_model("base")

# Set up Flask
app = Flask(__name__)
CORS(app)  # Allow CORS from any origin

# Ngrok tunnel for external access
public_url = ngrok.connect(5000)
print("🔗 Public ngrok URL:", public_url)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio_file uploaded"}), 400

    file = request.files['audio_file']
    filename = secure_filename(file.filename)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
            file.save(tmp.name)
            print(f"📁 Saved temp audio: {tmp.name}")

            # Transcribe using Whisper
            result = model.transcribe(tmp.name, language="ar")
            print("📋 Transcript:", result['text'])

        os.remove(tmp.name)  # Clean up temp file
        return jsonify({"transcript": result['text'].strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "Whisper Transcription Server is running."

# Start Flask
if __name__ == "__main__":
    app.run(port=5000)
