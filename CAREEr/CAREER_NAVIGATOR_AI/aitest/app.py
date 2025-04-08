import fitz  # PyMuPDF for text extraction
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Configure Gemini API Key
genai.configure(api_key="AIzaSyDl5f1aCkrMn8eK9oQ9kPQWs65hUekUE5U")


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF resume."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def generate_mcqs(resume_text):
    """Generates 15 MCQs testing both theoretical and practical knowledge."""
    prompt = f"""
    Analyze the following resume and create **15 multiple-choice questions (MCQs)**.
    
    **Rules:**
    - Each MCQ should have **four answer choices (A, B, C, D)**.
    - Ensure questions test **both theoretical and practical** knowledge.
    - **Half the questions should test fundamental concepts** (e.g., 'What is a tensor in TensorFlow?').
    - **Half the questions should test practical knowledge** (e.g., 'Which command initializes a TensorFlow model?').
    - Highlight the **correct answer**.

    **Example MCQs:**

    **1. Theoretical Question:**  
       What is a tensor in TensorFlow?  
       A) A variable  
       B) A multi-dimensional array  
       C) A function  
       D) A loss metric  

       **Correct Answer:** B) A multi-dimensional array  

    **2. Practical Question:**  
       Which TensorFlow function is used to create a sequential model?  
       A) tf.keras.layers.Model()  
       B) tf.keras.Sequential()  
       C) tf.create_model()  
       D) tf.build_model()  

       **Correct Answer:** B) tf.keras.Sequential()  

    **Now generate 15 such MCQs based on the following resume text:**
    
    Resume Text:
    {resume_text}
    """

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)

    mcqs = []
    if response:
        lines = response.text.strip().split("\n\n")  # Split MCQs
        for mcq in lines:
            parts = mcq.split("\n")
            if len(parts) >= 5:
                question = parts[0].replace("**Question:**", "").strip()
                options = [opt.strip() for opt in parts[1:5]]
                correct_answer = parts[-1].replace("**Correct Answer:**", "").strip()
                mcqs.append({"question": question, "options": options, "correct_answer": correct_answer})
    
    return mcqs


def assign_nft_tag(score):
    """Assigns NFT tag based on the candidate's score."""
    if score >= 90:
        return "Platinum NFT"
    elif score >= 70:
        return "Gold NFT"
    elif score >= 50:
        return "Silver NFT"
    else:
        return "No NFT"


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_resume():
    """Handles file upload and generates MCQs."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file uploaded"}), 400

    file_path = f"uploads/{file.filename}"
    file.save(file_path)
    resume_text = extract_text_from_pdf(file_path)

    mcqs = generate_mcqs(resume_text)
    return jsonify({"mcqs": mcqs})


@app.route("/evaluate", methods=["POST"])
def evaluate_answers():
    """Evaluates candidate MCQ answers and assigns an NFT tag."""
    data = request.json
    mcqs = data.get("mcqs", [])
    answers = data.get("answers", [])

    if not mcqs or not answers or len(mcqs) != len(answers):
        return jsonify({"error": "Invalid input"}), 400

    correct_count = sum(1 for i in range(len(mcqs)) if mcqs[i]["correct_answer"].startswith(answers[i]))
    score = (correct_count / len(mcqs)) * 100
    nft_tag = assign_nft_tag(score)

    return jsonify({"score": f"{score:.2f}%", "nft_tag": nft_tag})


if __name__ == "__main__":
    app.run(debug=True,port=9000)
