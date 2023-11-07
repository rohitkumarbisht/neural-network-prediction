from flask import request, make_response
from flask_restful import Resource
from psycopg2.extensions import AsIs, register_adapter
import pickle as pkl
import io
import pandas as pd
import numpy as np
import json
import psycopg2
import config


class PredictionFileUpload(Resource):
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
                    column_data_types = {
                        col_name: data_type for col_name, data_type in cursor.fetchall()}
                    return column_data_types
        except Exception as e:
            return f"Error fetching column data types:{e}"

    def cast_dataframe_to_db_data_types(self, column_data_types, dataframe):
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

    def save_result_to_postgresql(self, result_df,):
        num_col = result_df.shape[1]
        col_names = result_df.columns.tolist()
        cols_required = ",".join(['"'+col + '"' for col in col_names])
        val_type = (',').join(['%s'] * num_col)
        column_data_types = self.get_column_data_types()
        type_val_tuple = self.cast_dataframe_to_db_data_types(
            column_data_types, result_df)
        val_tuple = []
        for index, row in type_val_tuple.iterrows():
            val_tuple.append(tuple(row))
        try:
            with psycopg2.connect(
                self.connection_string
            ) as conn:
                with conn.cursor() as cursor:
                    sql = f"""INSERT INTO {config.db_name}.{config.schema_name}.{config.table_name} ({cols_required}) VALUES ({val_type});"""
                    for val in val_tuple:
                        cursor.execute(sql, val)
            conn.commit()
            return 'Data saved to the database successfully'
        except Exception as e:
            return f"Failed to save training results to the database: {e}"

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
            success = self.save_result_to_postgresql(result_df)
            response_data = {
                "message": "Prediction completed successfully",
                "status": success,
                "result": result_df.to_dict(orient='records')
            }

            # Serialize the dictionary to a JSON string with indentation
            result_json = json.dumps(response_data, indent=4)

            return make_response(result_json, 200, {"Content-Type": "application/json"})
        except pd.errors.ParserError as e:
            return make_response({"error": f"Error parsing CSV data: {str(e)}"}, 406)
