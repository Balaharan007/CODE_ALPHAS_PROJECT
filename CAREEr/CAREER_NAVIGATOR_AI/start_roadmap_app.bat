@echo off
echo Starting Career Roadmap Generator...
cd C:\Users\haran\OneDrive\Desktop\CAREER_NAVIGATOR_AI\roadmapGen
start /min cmd /c "python app.py"
echo Application started! You can now access it at http://127.0.0.1:5000
timeout /t 2
start http://127.0.0.1:5000
