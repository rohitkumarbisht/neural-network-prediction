# Create a Virtual Environment
python -m venv venv

# Activate the Virtual Environment:
venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt

# To run
python app.py

# If any new package is installed use the following commanf to save that in requirements.txt
pip freeze > requirements.txt