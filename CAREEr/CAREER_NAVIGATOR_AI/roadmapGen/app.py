from flask import Flask, render_template, request
import google.generativeai as genai
import pandas as pd
from rapidfuzz import process
import os
from config import API_KEY

app = Flask(__name__)

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Load Course Dataset with Error Handling
COURSE_DATA_PATH = "data/courses.csv"

try:
    courses_df = pd.read_csv(COURSE_DATA_PATH, encoding="utf-8", on_bad_lines="skip")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    courses_df = pd.DataFrame(columns=["Course Name", "Description", "Link", "Rating", "Enrollments"])  # Empty fallback

def generate_roadmap(role):
    """Generate a structured learning roadmap using Gemini AI."""
    prompt = f"""
    Generate a structured learning roadmap to become a {role}. 
    The roadmap should have major milestones separated by ' ➝ '. 
    Example: 'Starting Point ➝ Statistics ➝ Python ➝ SQL ➝ Excel ➝ Data Visualization ➝ BI Tools ➝ Being Awesome!'
    """
    response = model.generate_content(prompt)

    return response.text.strip() if hasattr(response, 'text') else "❌ Invalid response from AI."

def get_top_courses(step, top_n=5, threshold=60):
    """Get the top N trending courses based on fuzzy matching."""
    if courses_df.empty:
        return []

    # Find top N best matches using RapidFuzz
    matches = process.extract(step, courses_df["Course Name"].dropna().tolist(), limit=top_n)

    # Filter based on match quality
    matched_courses = [match[0] for match in matches if match[1] >= threshold]

    if matched_courses:
        filtered_df = courses_df[courses_df["Course Name"].isin(matched_courses)]

        # Sort courses by rating and enrollments if available
        if "Rating" in filtered_df.columns and "Enrollments" in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by=["Rating", "Enrollments"], ascending=[False, False])

        return filtered_df.head(top_n).to_dict(orient="records")

    return []

@app.route("/", methods=["GET", "POST"])
def index():
    roadmap = None
    if request.method == "POST":
        role = request.form.get("role")
        roadmap = generate_roadmap(role)

    return render_template("index.html", roadmap=roadmap)

@app.route("/roadmap/<step>")
def roadmap_step(step):
    """Render a detailed explanation page for a roadmap step with top recommended courses."""
    try:
        recommended_courses = get_top_courses(step, top_n=5, threshold=50)  # Lower threshold to find more matches
        roadmap = request.args.get("roadmap", "")
        if isinstance(roadmap, str):
            steps = roadmap.split(" ➝ ")
        else:
            steps = []
        return render_template("roadmap_step.html", step=step, recommended_courses=recommended_courses, steps=steps, roadmap=roadmap)
    except Exception as e:
        print(f"Error retrieving courses: {e}")
        # Return with empty recommended courses
        roadmap = request.args.get("roadmap", "")
        if isinstance(roadmap, str):
            steps = roadmap.split(" ➝ ")
        else:
            steps = []
        return render_template("roadmap_step.html", step=step, recommended_courses=[], steps=steps, roadmap=roadmap, error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Use default port
