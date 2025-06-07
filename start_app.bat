@echo off
echo.
echo =====================================================
echo         🎬 YouTube Summarizer
echo =====================================================
echo.
echo 📁 Project Directory: %CD%
echo 🐍 Python Version:
python --version
echo.
echo 📦 Verifying Dependencies...
python -c "import streamlit; print('✅ Streamlit:', streamlit.__version__)" 2>nul || echo "❌ Streamlit not found - run: pip install -r requirements.txt"
echo.
echo 🌐 Starting application at: http://localhost:8501
echo 📝 Press Ctrl+C to stop the server
echo.
echo =====================================================
streamlit run streamlit_app.py --server.port 8501 --server.headless true
echo.
echo Application stopped.
pause
