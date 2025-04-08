from flask import Flask, render_template, request
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load dataset
file_path = r"C:\Users\haran\OneDrive\Desktop\TECHTEAM\IEEE\CAREER_NAVIGATOR_AI\jobRecommendationSystem\job_listings.csv"
df = pd.read_csv(file_path)

def recommend_jobs(search_role):
    search_role = search_role.lower()  # Convert input to lowercase for better matching

    # Filter jobs where any keyword in the search exists in the Job Title
    filtered_jobs = df[df['Job Title'].str.lower().str.contains(search_role, na=False)]
    
    # Sort jobs by rating (descending)
    recommended_jobs = filtered_jobs.sort_values(by='Rating', ascending=False)
    
    return recommended_jobs.to_dict(orient='records')

@app.route('/', methods=['GET', 'POST'])
def home():
    jobs = []
    if request.method == 'POST':
        search_role = request.form.get('job_role')
        jobs = recommend_jobs(search_role)
    return render_template('index.html', jobs=jobs)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
