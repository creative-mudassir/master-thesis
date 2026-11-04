@echo off
rem Overnight 50 km run: cube (03) then labels (04). Started by Windows Task Scheduler (task "edc_50km"),
rem so it keeps running when the terminal / VS Code is closed. Log: data\logs\run_50km.log
rem Check progress:  powershell -c "Get-Content 'data\logs\run_50km.log' -Tail 25"
rem If 03 failed or was stopped: run this file again; finished months/parts are skipped (no --overwrite).

cd /d "%~dp0.."
if not exist data\logs mkdir data\logs
set LOG=data\logs\run_50km.log
set CONDA=%USERPROFILE%\miniconda3\condabin\conda.bat

echo ===== %DATE% %TIME% START 03_build_cube --size-km 50 ===== >> %LOG%
call "%CONDA%" run -n edc --no-capture-output python -u scripts\03_build_cube.py --size-km 50 >> %LOG% 2>&1
echo ===== %DATE% %TIME% END 03_build_cube exit %ERRORLEVEL% ===== >> %LOG%

echo ===== %DATE% %TIME% START 04_make_labels --size-km 50 ===== >> %LOG%
call "%CONDA%" run -n edc --no-capture-output python -u scripts\04_make_labels.py --size-km 50 >> %LOG% 2>&1
echo ===== %DATE% %TIME% END 04_make_labels exit %ERRORLEVEL% ===== >> %LOG%
echo ===== ALL DONE ===== >> %LOG%
