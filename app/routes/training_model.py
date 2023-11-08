from flask import make_response, Response
from flask_restful import Resource
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import contextlib
import config
import psycopg2
import time
from datetime import date
from app.routes.distribution_graph import DistributionGraph
from app.utils.file_open import read_file, open_model

class TrainingModel(Resource):
    @contextlib.contextmanager
    def open_file(self, filename, mode):
        try:
            with open(filename, mode) as file:
                yield file
        except FileNotFoundError as e:
            return make_response({"error": f"File not found: {e.filename}"}, 404)

    def train_model(self, X, y):
        MPL_model = MLPClassifier(max_iter=20)
        MPL_model.fit(X, y)
        return MPL_model

    def save_training_results_to_database(self, accuracy, precision, training_time, date_modified):
        try:
            with psycopg2.connect(
                dbname=config.db_name, user=config.db_user, password=config.db_password, host=config.db_host, port=config.db_port
            ) as conn:
                with conn.cursor() as cursor:
                    sql = f"INSERT INTO {config.schema_name}.{config.config_table} (accuracy, precision, training_time, date_modified) VALUES (%s, %s, %s, %s);"
                    cursor.execute(sql, (accuracy, precision,
                                   training_time, date_modified))
                    conn.commit()
        except Exception as e:
            return make_response({"error": f"Failed to save training results to the database: {e}"}, 500)

    def post(self):
        csv_data_instance = DistributionGraph()
        csv_data = csv_data_instance.fetch_csv_data()
        actual_columns = read_file("highly_correlated_columns.txt", "r")
        if isinstance(actual_columns, Response):
            return actual_columns

        selected_column = read_file("target_column.txt", "r")
        if isinstance(selected_column, Response):
            return selected_column

        X = csv_data[actual_columns]
        y = csv_data[selected_column]
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Start measuring the training time
        start_time = time.time()
        #  Train the Neural Network model
        MPL_model = self.train_model(X, y)
        end_time = time.time()
        # Calculate the training time
        training_time = end_time - start_time
        # calculate accuracy, precision & modified_on
        accuracy = cross_val_score(
            MPL_model, X, y, cv=cv, scoring='accuracy').mean()
        precision = cross_val_score(
            MPL_model, X, y, cv=cv, scoring='precision_macro').mean()
        today = date.today()
        modified_on = today.isoformat()
        # Save the model to file
        open_model('mpl_model.pkl','wb',MPL_model)
        # save training results to database
        result = self.save_training_results_to_database(
            accuracy, precision, training_time, modified_on)
        if result:
            return result

        return make_response({"message": "Model training completed successfully", 'accuracy': accuracy, 'precision': precision, 'training_time': training_time, "date_modified": modified_on}, 200)