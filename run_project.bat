@echo off
echo Starting Pneumonia Detection Project...

REM Activate your existing TF environment
call conda activate tf

REM Install project-specific requirements
pip install -r requirements_project.txt

REM Run GPU optimization test
python src/gpu_optimizer.py

REM Start the main training
python main.py

pause