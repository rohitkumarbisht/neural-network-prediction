from flask import request,make_response
from flask_restful import Resource
import io
import pandas as pd

class UploadCSV(Resource):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UploadCSV, cls).__new__(cls)
            cls._instance.uploaded_data = None
        return cls._instance
    
    def fetch_data(self):
        if self.uploaded_data is None:
            return None
        else:
            return self.uploaded_data

    def post(self):
        file = request.get_data()
        if not file:
            return make_response({"error": "No file was uploaded"}, 400)
        try:
            binary_io_train = io.BytesIO(file)
            uploaded_csv = pd.read_csv(binary_io_train)
            label_mapping = {'Dropout': 0, 'Graduate': 1, 'Enrolled': 2}
            uploaded_csv['Target'] = uploaded_csv['Target'].map(label_mapping)
            self.uploaded_data = uploaded_csv
            return make_response({"message": "CSV data uploaded successfully"}, 201)
        except pd.errors.ParserError as e:
            return make_response({"error": f"Error parsing CSV data: {str(e)}"}, 400)
