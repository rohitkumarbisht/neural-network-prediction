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

##ABOUT:
The project focus on training the neural network based model with appropriate data and predicting the chances of a student's dropout based on various relevant parameters.
