from flask import request, make_response
from flask_restful import Resource
from psycopg2.extensions import AsIs,register_adapter
from app.utils.file_open import read_file, open_model
import pandas as pd
import numpy as np
import json
import psycopg2
import config

class PredictionInput(Resource):
    def __init__(self):
        self.connection_string = self.build_connection_string()

    def build_connection_string(self):
        return f"dbname={config.db_name} user={config.db_user} password={config.db_password} host={config.db_host} port={config.db_port}"
    
    def get_column_data_types(self):
        try:
            with psycopg2.connect(
                self.connection_string
            ) as conn:
                with conn.cursor() as cursor:
                    sql = f"""SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{config.table_name}';"""
                    cursor.execute(sql)
                    column_data_types = {col_name: data_type for col_name, data_type in cursor.fetchall()}
                    return column_data_types
        except Exception as e:
            return f"Error fetching column data types:{e}"
    
    def cast_dataframe_to_db_data_types(self,column_data_types,dataframe):
        for column in dataframe.columns:
        # Check if the column is found in the PostgreSQL data types dictionary
            if column in column_data_types:
                data_type = column_data_types[column]
                # Dynamically cast the column to the appropriate data type
                if data_type == 'integer':
                    register_adapter(np.int64, AsIs)
                    dataframe[column] = dataframe[column].astype(np.int64)
                elif data_type == 'real':
                    dataframe[column] = dataframe[column].astype(float)
                elif data_type == 'character varying':
                    dataframe[column] = dataframe[column].astype(str)
        return dataframe
    

    def save_result_to_postgresql(self,result_df,):
        num_col = result_df.shape[1]
        col_names = result_df.columns.tolist()
        cols_required = ",".join(['"'+col + '"' for col in col_names])
        val_type = (',').join(['%s'] * num_col)
        column_data_types= self.get_column_data_types()
        type_val_tuple = self.cast_dataframe_to_db_data_types(column_data_types,result_df)
        val_tuple = type_val_tuple.iloc[-1]
        try:
            with psycopg2.connect(
                self.connection_string
            ) as conn:
                with conn.cursor() as cursor:
                    sql = f"""INSERT INTO {config.db_name}.{config.schema_name}.{config.table_name} ({cols_required}) VALUES ({val_type});"""
                    cursor.execute(sql, val_tuple)
            conn.commit()
            return 'Data saved to the database successfully'
        except Exception as e:
            return f"Failed to save training results to the database: {e}"

    def post(self):
        # Load the model
        try:
            model_pkl = open_model('mpl_model.pkl', 'rb')
        except FileNotFoundError:
            return make_response({'error': 'Model file not found, Please train the model!'}, 404)

        # Read input values from the request body as JSON
        input_data = request.get_json()

        if input_data is None:
            return make_response({"error": "No input data provided in the request body"}, 400)

        actual_columns = read_file("highly_correlated_columns.txt", "r")

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
            return make_response({"error": "Prediction failed as data can't be processed"},422)

        # Convert y_pred values to labels
        y_pred_labels = ['Dropout' if x == 0 else (
            'Graduate' if x == 1 else 'Enrolled') for x in y_pred]
        # Create DF combining input data and prediction result
        result_df = input_data_df[actual_columns].copy()
        result_df['target'] = y_pred_labels
        success = self.save_result_to_postgresql(result_df)
        response_data = {
            "message": "Prediction completed successfully",
            "status" : success,
            "result": result_df.to_dict(orient='records')
        }
        # Serialize the dictionary to a JSON string with indentation
        result_json = json.dumps(response_data, indent=4)
        return make_response(result_json, 200, {"Content-Type": "application/json"})

