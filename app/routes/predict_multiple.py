from flask import request, make_response
from flask_restful import Resource
import pickle as pkl
import io
import pandas as pd
import json

class PredictionFileUpload(Resource):
    def post(self):
        try:
            with open('mpl_model.pkl', 'rb') as file:
                model_pkl = pkl.load(file)
        except FileNotFoundError:
            return make_response({'error': 'Model file not found, Please train the model!'}, 404)

        data = request.get_data()
        if not data:
            return make_response({"error": "No file was uploaded"}, 400)
        try:
            binary_io = io.BytesIO(data)
            df = pd.read_csv(binary_io)

            with open("highly_correlated_columns.txt", "r") as file:
                actual_columns = file.read().splitlines()

            pred_X = df[actual_columns]
            y_pred = model_pkl.predict(pred_X)
            y_pred_labels = ['Dropout' if x == 0 else (
                'Graduate' if x == 1 else 'Enrolled') for x in y_pred]

            result_df = pred_X.copy()
            result_df['target'] = y_pred_labels
            response_data = {
                "message": "Prediction completed successfully",
                "result": result_df.to_dict(orient='records')
            }

            # Serialize the dictionary to a JSON string with indentation
            result_json = json.dumps(response_data, indent=4)

            return make_response(result_json, 200, {"Content-Type": "application/json"})
        except pd.errors.ParserError as e:
            return make_response({"error": f"Error parsing CSV data: {str(e)}"}, 406)
