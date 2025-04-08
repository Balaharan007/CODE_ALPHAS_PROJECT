@echo off
echo ===== Starting Career Navigator AI Backend =====

echo Checking MongoDB status...
mongod --version >nul 2>&1
if %errorlevel% neq 0 (
    echo MongoDB is not installed or not in PATH.
    echo Please install MongoDB from https://www.mongodb.com/try/download/community
    echo.
    echo As a temporary solution, we'll use MongoDB Atlas cloud database.
) else (
    echo MongoDB is installed.
    echo Starting MongoDB service...
    start /B mongod --dbpath=%USERPROFILE%\.mongodb\data\db
    if not exist %USERPROFILE%\.mongodb\data\db mkdir %USERPROFILE%\.mongodb\data\db
    timeout /t 5
)

echo Starting Node.js server...
cd /d %~dp0
node server.js

pause 