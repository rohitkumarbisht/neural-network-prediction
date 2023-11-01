from flask import request, make_response
from flask_restful import Resource
import pickle as pkl
import pandas as pd
import json
import psycopg2
import config

class PredictionInput(Resource):
    def save_result_to_postgresql(self,result_df):
        num_col = result_df.shape[1]
        col_names = result_df.columns.tolist()
        cols_required = ",".join(['"'+col + '"' for col in col_names])
        val_type = (',').join(['%s'] * num_col)
        val_tuple = ()
        for i in range(num_col):
            val = result_df.iloc[-1,i]
            val_tuple += (val,)
        try:
            with psycopg2.connect(
                dbname=config.db_name, user=config.db_user, password=config.db_password, host=config.db_host, port=config.db_port
            ) as conn:
                with conn.cursor() as cursor:
                    sql = f"INSERT INTO {config.schema_name}.{config.table_name} ({cols_required}) VALUES ({val_type});"
                    cursor.execute(sql, val_tuple)
                    conn.commit()
                    cursor.close()
                    conn.close()
        except Exception as e:
            return make_response({"error": f"Failed to save training results to the database: {e}"}, 500)

    def post(self):
        # Load the model
        try:
            with open('mpl_model.pkl', 'rb') as file:
                model_pkl = pkl.load(file)
        except FileNotFoundError:
            error = 'Model file not found, Please train the model!'
            return make_response({'error': error}, 404)

        # Read input values from the request body as JSON
        input_data = request.get_json()

        if input_data is None:
            return make_response({"error": "No input data provided in the request body"}, 400)

        # Ensure that the input data keys match the columns in "highly_correlated_columns.txt"
        with open("highly_correlated_columns.txt", "r") as file:
            actual_columns = file.read().splitlines()
            missing_columns = [
                column for column in actual_columns if column not in input_data]

        if missing_columns:
            return make_response({"error": f"Input data missing for column:{', '.join(missing_columns)}"}, 500)

        # Create a DataFrame with input values
        input_data_df = pd.DataFrame([input_data])
        pred_X = input_data_df[actual_columns]
        # Predict using the model
        y_pred = model_pkl.predict(pred_X)
        if len(y_pred) == 0:
            return make_response({"error": "Prediction failed"}, 500)

        # Convert y_pred values to labels
        y_pred_labels = ['Dropout' if x == 0 else (
            'Graduate' if x == 1 else 'Enrolled') for x in y_pred]
        # Create DF combining input data and prediction result
        result_df = input_data_df[actual_columns].copy()
        result_df['target'] = y_pred_labels
        response_data = {
            "message": "Prediction completed successfully",
            "result": result_df.to_dict(orient='records')
        }
        self.save_result_to_postgresql(result_df)
        # Serialize the dictionary to a JSON string with indentation
        result_json = json.dumps(response_data, indent=4)
        return make_response(result_json, 200, {"Content-Type": "application/json"})

