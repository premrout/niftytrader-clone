@echo off
title NiftyTrader App + Ngrok

echo Starting Streamlit app...
start cmd /k "cd /d %~dp0 && python -m streamlit run app.py"

echo Waiting 5 seconds for Streamlit to start...
timeout /t 5 >nul

echo Starting ngrok tunnel...
start cmd /k "cd C:\Users\premr\ngrok && ngrok.exe http 8501"

echo -----------------------------------------------------
echo Your Streamlit app is now running locally at:
echo     http://localhost:8501
echo.
echo Ngrok will provide a public URL shortly.
echo -----------------------------------------------------

pause
