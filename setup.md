# Setup and Running Guide

This guide provides step-by-step instructions to run the Brazil Cities Dashboard application.

## Prerequisites Validation

Before starting, verify that you have Python installed and accessible:

```bash
python --version
```

Expected output: Python 3.8 or higher

## Step-by-Step Setup

### 1. Navigate to Project Directory

```bash
cd "C:\path\to\project\directory\"
```

### 2. Create Virtual Environment (Recommended)

_Note: Ignore this step if you have Anaconda installed_

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

_Note: Ignore this step if you have Anaconda installed_

**Windows (PowerShell):**

```bash
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```bash
.\venv\Scripts\activate.bat
```

### 4. Install Required Dependencies

```bash
pip install shiny shinywidgets pandas numpy matplotlib plotly scikit-learn
```

### 5. Verify Data Files

Ensure the following files exist in the `data/` folder:

- `BRAZIL_CITIES_CLEANED.csv`
- `brazil-states.geojson`

### 6. Run the Dashboard

Navigate to the `src` directory and start the application:

```bash
cd src
python -m shiny run app.py --port 8000
```

### 7. Access the Dashboard

Open your web browser and go to:

```
http://localhost:8000
```

## Common Errors and Solutions

### Error: "ModuleNotFoundError: No module named 'shiny'"

**Solution:** Install the missing package:

```bash
pip install shiny
```

### Error: "Python is not recognized as an internal or external command"

**Solution:** Python is not in your PATH. Either:

- Add Python to your system PATH
- Use the full path to python.exe (e.g., `C:\Python39\python.exe`)

### Error: "FileNotFoundError: [Errno 2] No such file or directory"

**Solution:** Ensure you're running the command from the correct directory. The app.py file should be in the `src` folder.

### Error: "Address already in use"

**Solution:** Port 8000 is already in use. Either:

- Stop the other application using port 8000
- Use a different port: `python -m shiny run app.py --port 8001`

### Error: "Permission denied" when activating virtual environment

**Solution:** Run PowerShell as Administrator and execute:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Error: Joblib warnings on Windows

**Solution:** These warnings are already suppressed in the code. If they persist, ignore them as they don't affect functionality.

## Quick Start (Without Virtual Environment)

If you prefer not to use a virtual environment:

```bash
cd "E:\DkIT\__Data Visualisation\CA_Main_Project\Repo\src"
python -m shiny run app.py --port 8000
```

## Stopping the Application

Press `Ctrl + C` in the terminal to stop the dashboard server.
