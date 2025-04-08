import os
import cv2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load Haar Cascade Path from .env
HAAR_CASCADE_PATH = os.getenv("HAAR_CASCADE_PATH")

# Verify if the file exists
if not os.path.exists(HAAR_CASCADE_PATH):
    print(f"❌ Haar Cascade XML file not found at {HAAR_CASCADE_PATH}")
    exit()

# Load the classifier
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Check if cascade loaded successfully
if face_cascade.empty():
    print("❌ Error: Haar Cascade failed to load. Check the XML file path.")
    exit()
else:
    print("✅ Haar Cascade loaded successfully!")
