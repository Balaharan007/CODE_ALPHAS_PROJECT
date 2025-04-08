from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Gemini AI
genai.configure(api_key="AIzaSyAurjcaQWypq4YXFxiIHwzdWzcyL1Y_P2Y")

# Use the latest available model
model = genai.GenerativeModel("gemini-1.5-pro-latest")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
 
    if not user_message:
        return jsonify({"response": "Please enter a message."})

    # Generate AI response
    response = model.generate_content(user_message)
    return jsonify({"response": response.text})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
