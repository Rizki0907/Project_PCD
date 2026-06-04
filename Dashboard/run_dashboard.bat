@echo off
echo =========================================
echo  CastGuard AI -- Defect Detection Dashboard
echo  PCD Project -- Unesa 2024
echo =========================================
echo.
echo Starting Streamlit dashboard...
echo Open browser at: http://localhost:8501
echo.
cd /d "%~dp0"
streamlit run app.py --server.port 8501 --server.headless false
pause
