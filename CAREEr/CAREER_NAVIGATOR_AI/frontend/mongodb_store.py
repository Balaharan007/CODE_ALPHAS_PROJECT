# import pandas as pd
# from pymongo import MongoClient

# # Load the CSV file
# file_path = r"C:\Users\haran\OneDrive\Desktop\CAREEr\CAREER_NAVIGATOR_AI\frontend\user_information.csv"
# data = pd.read_csv(file_path)

# # Connect to MongoDB
# client = MongoClient("mongodb://localhost:27017/")

# # Create or connect to userAuthDB
# db = client["userAuthDB"]

# # Create or connect to user_information collection
# collection = db["leaderboard"]

# # Convert data to dictionary format for MongoDB
# data_dict = data.to_dict(orient="records")

# # Insert data into the collection
# collection.insert_many(data_dict)

# print("Data successfully inserted into user_information collection.")
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")

# Create or connect to userAuthDB database
db = client["userAuthDB"]

# Create or connect to users collection
collection = db["user_information"]

# Define the data as a dictionary
user_data = {
    "Name": "Bala",
    "Role": "Full Stack Developer",
    "Relevant Course": "Web Development with React & Node",
    "Certification": "Yes",
    "NFT Verification": "Yes",
    "NFT Tier": "Platinum",
    "Experience": "2 years",
    "Portfolio/GitHub": "https://github.com/bala",
    "Education": "Bachelor's",
    "University": "Harvard University",
    "Skills & Performance": {
        "Technical Skill Score": 88,
        "Soft Skill Score": 75,
        "Overall Assessment Score": 89
    },
    "Course Completion": "5%",
    "Skill Level": "Intermediate",
    "Total Progress": "90%",
    "Total EP": 12521,
    "Badges Earned": 4,
    "Current Streak": "51 days",
    "Total Days Activated": 154,
    "Rank": "#77",
    "Time Spent": "751 hrs",
    "Course Learning Areas": [
        "Machine Learning",
        "Data Science",
        "Full Stack Development"
    ],
    "Leaderboard": "Active",
    "AI Test": "Available",
    "Welcome Message": "Welcome, Bala!",
    "Overall Progress": "90%",
    "Current Streak": "51 days",
    "Learning Time": "751 hours",
    "Badges Earned Summary": 4,
    "Email": "balaharan@gmail.com"
}

# Insert the document into the collection
insert_result = collection.insert_one(user_data)

# Print confirmation with inserted ID
print(f"Data inserted successfully with ID: {insert_result.inserted_id}")
