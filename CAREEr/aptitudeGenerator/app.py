from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from flask_cors import CORS
import json
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Gemini AI  
API_KEY = "AIzaSyA0-FQqUYvZTtDAS0aqprQf6Q0Y6Bred10"
genai.configure(api_key=API_KEY)

# Use gemini-pro as it's more stable
model = genai.GenerativeModel("gemini-pro")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-aptitude', methods=["POST"])
def generate_aptitude():
    data = request.json
    count = data.get('count', 10)  # Default to 10 questions if not specified
    
    prompt = f"""
    Generate {count} aptitude test questions for a job interview. 
    Each question should:
    1. Be multiple choice with 4 options
    2. Cover a mix of logical reasoning, numerical ability, and verbal reasoning
    3. Be appropriate for technical job candidates
    4. Include the correct answer
    
    Format the response as a JSON array of objects with the following structure:
    [
      {{
        "question": "The question text",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correctAnswer": 0  // Index of the correct option (0-based)
      }},
      // more questions...
    ]
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Extract the JSON part from the response
        # Find JSON content between ```json and ``` if present
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Otherwise try to find anything that looks like a JSON array
            json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text
        
        # Clean up the JSON string to ensure it's valid
        # Remove any literal newlines and clean up the string
        json_str = json_str.strip()
        
        # Parse the JSON
        questions = json.loads(json_str)
        
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e), "response": response_text if 'response' in locals() else "No response generated"}), 500

if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    import os
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    # Create index.html if it doesn't exist
    index_html = os.path.join(templates_dir, "index.html")
    if not os.path.exists(index_html):
        with open(index_html, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aptitude Test Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input, button {
            padding: 8px;
            font-size: 16px;
        }
        button {
            background-color: #4285F4;
            color: white;
            border: none;
            cursor: pointer;
            padding: 10px 15px;
        }
        button:hover {
            background-color: #2b6ed9;
        }
        #questions {
            margin-top: 20px;
        }
        .question {
            background-color: #f5f5f5;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
        }
        .options {
            margin-top: 10px;
            margin-left: 20px;
        }
        .correct {
            color: green;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Aptitude Test Generator</h1>
    
    <div class="form-group">
        <label for="questionCount">Number of Questions:</label>
        <input type="number" id="questionCount" min="1" max="20" value="10">
        <button id="generateBtn">Generate Questions</button>
    </div>
    
    <div id="questions"></div>
    
    <script>
        document.getElementById('generateBtn').addEventListener('click', function() {
            const count = document.getElementById('questionCount').value;
            const questionsDiv = document.getElementById('questions');
            
            questionsDiv.innerHTML = '<p>Generating questions...</p>';
            
            fetch('/generate-aptitude', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ count: parseInt(count) })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    questionsDiv.innerHTML = '<p>Error: ' + data.error + '</p>';
                    return;
                }
                
                questionsDiv.innerHTML = '';
                data.questions.forEach((q, index) => {
                    const questionDiv = document.createElement('div');
                    questionDiv.className = 'question';
                    
                    questionDiv.innerHTML = `
                        <h3>Question ${index + 1}</h3>
                        <p>${q.question}</p>
                        <div class="options">
                            ${q.options.map((option, i) => `
                                <div>
                                    <input type="radio" id="q${index}_o${i}" name="q${index}" ${i === q.correctAnswer ? 'checked' : ''}>
                                    <label for="q${index}_o${i}" ${i === q.correctAnswer ? 'class="correct"' : ''}>${option}</label>
                                </div>
                            `).join('')}
                        </div>
                    `;
                    
                    questionsDiv.appendChild(questionDiv);
                });
            })
            .catch(error => {
                questionsDiv.innerHTML = '<p>Error: ' + error.message + '</p>';
            });
        });
    </script>
</body>
</html>
            """)
    
    print("Aptitude Generator running on http://127.0.0.1:5004")
    app.run(debug=True, port=5004)
