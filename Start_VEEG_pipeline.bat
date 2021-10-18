@echo on
call C:\Users\rudik\miniconda3\Scripts\activate.bat
call conda activate mne023_4
cd .\src\
jupyter notebook VEEG.ipynb
pause
