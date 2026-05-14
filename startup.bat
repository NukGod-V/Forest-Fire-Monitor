@echo off
echo 🔥 Starting Forest Fire Monitoring System...

:: Check if .env file exists
if not exist .env (
    echo ❌ .env file not found!
    echo Creating sample .env file...
    (
        echo # NASA FIRMS API Configuration
        echo NASA_FIRMS_API_KEY=your_api_key_here
        echo # API Configuration
        echo API_HOST=0.0.0.0
        echo API_PORT=8000
        echo API_DEBUG=True
        echo # Cache Configuration
        echo CACHE_TTL=300
        echo CACHE_MAXSIZE=100
    ) > .env
    echo ✅ Sample .env file created. Please update with your actual API key.
)

:: Check if virtual environment exists
if not exist venv (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment and install requirements
echo 🔧 Activating virtual environment and installing packages...
call venv\Scripts\activate
pip install -r requirements.txt

:: Start FastAPI server in a new window
echo 🚀 Starting FastAPI server in a new window...
start "FastAPI_Server" cmd /c "venv\Scripts\activate && uvicorn main:app --host 0.0.0.0 --port 8000"

echo ⏳ Waiting for API server to start...
timeout /t 5 > nul

:: Start Streamlit app
echo 🎨 Starting Streamlit app...
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo ✅ System is running. Close this window to stop the Streamlit app.