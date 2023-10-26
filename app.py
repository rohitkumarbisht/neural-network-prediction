from flask import Flask, request, make_response
from flask_restful import Resource, Api
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle as pkl
import os
import io
from flask_swagger_ui import get_swaggerui_blueprint
app = Flask(__name__)
api = Api(app)
import psycopg2

# Define your PostgreSQL database connection settings
db_host = 'airbyte.cqqg4q5hnscs.ap-south-1.rds.amazonaws.com'
db_port = 5432
db_user = 'airbyte'
db_password = 'F648d&lTHltriVidRa0R'
db_name = 'learninganalytics'
schema_name = 'learninganalytics'
table_name = "dropout_data"

# swagger config
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGER_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL, API_URL,
    config = {
        'app_name': 'Dropout Prediction'
    }
)

app.register_blueprint(SWAGGER_BLUEPRINT, specs_url = SWAGGER_URL)

# connect with the PostgreSQL Server
def fetch_data_from_postgresql():
    global df
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = conn.cursor()

        # Fetch all data from the table
        cursor.execute(f'SELECT * FROM {db_name}.{schema_name}."{table_name}"')
        data = cursor.fetchall()

        # Get column names from the table
        cursor.execute(
            'SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s', (schema_name, table_name,))
        columns = [col[0] for col in cursor.fetchall()]

        df = pd.DataFrame(data, columns=columns)

        cursor.close()
        conn.close()

        return df

    except Exception as e:
        print(f"Error fetching data from PostgreSQL: {str(e)}")
        return None

# home page


class Home(Resource):
    def get(self):
        return {"message": "Welcome to the API"}


api.add_resource(Home, '/')

# upload training data


class UploadCSV(Resource):
    def post(self):
        global csv_data
        file = request.get_data()
        if not file:
            return make_response({"error": "No file was uploaded"}, 400)
        try:
            binary_io_train = io.BytesIO(file)
            csv_data = pd.read_csv(binary_io_train)
            label_mapping = {'Dropout': 0, 'Graduate': 1, 'Enrolled': 2}
            csv_data['Target'] = csv_data['Target'].map(label_mapping)
            return {"message": "CSV data uploaded successfully"}
        except pd.errors.ParserError as e:
            return make_response({"error": f"Error parsing CSV data: {str(e)}"}, 400)



api.add_resource(UploadCSV, '/upload-csv')

# Distribution Graph Page


class DistributionGraph(Resource):
    def get(self):
        csv_data = fetch_data_from_postgresql()
        csv_data = csv_data.iloc[:-1]
        csv_data = csv_data.drop('student_id', axis=1)  
        label_mapping = {
            'Dropout': 0,
            'Graduate': 1,
            'Enrolled': 2
        }
        csv_data['target'] = csv_data['target'].map(label_mapping)
        if csv_data is not None:
            # Your distribution graph logic here
            num_columns = len(csv_data.columns)
            num_rows = (num_columns + 4) // 5
            num_columns_last_row = num_columns % 5

            # Create a grid of subplots
            fig, axes = plt.subplots(
                nrows=num_rows, ncols=5, figsize=(15, 2 * num_rows))

            # Plot histograms for each column
            for i, col in enumerate(csv_data.columns):
                row_index = i // 5
                col_index = i % 5
                ax = axes[row_index, col_index]
                if i < num_columns:
                    ax.hist(csv_data[col], bins=10)
                    ax.set_title(col)
                else:
                    ax.axis('off')

            if num_columns_last_row > 0:
                for j in range(num_columns_last_row, 5):
                    axes[num_rows - 1, j].axis('off')
            plt.tight_layout()
            png_path = 'static/images/distribution.png'
            plt.savefig(png_path)
            if os.path.exists(png_path):
                # Assuming successful processing, return a 200 OK status code
                # 200 OK
                return make_response({"message": "Distribution graph generated", "png_path": png_path}, 200)
            else:
                # If the file does not exist, return a 404 Not Found status code
                # 404 Not Found
                return make_response({"error": "Distribution graph not found"}, 404)

        # If no CSV data is available,  return a 403 Forbidden Error status code
        else:
            # 403 Forbidden
            return make_response({"error": "No data available"}, 403)


api.add_resource(DistributionGraph, '/distribution-graph')

# correlation graph page


class CorrelationGraph(Resource):
    def post(self):
        global actual_columns, selected_column
        selected_column = request.args.get("selected_column")
        if not selected_column:
            error = 'No target column was selected'
            return {"error": error}

        if selected_column:
            correlation_with_Target = csv_data.corr()[selected_column]
            correlation_with_Target = correlation_with_Target.drop(
                selected_column)
            correlation_with_Target_sorted = correlation_with_Target.sort_values(
                ascending=False)

            # Plot the histogram
            plt.figure(figsize=(10, 6))
            bars = plt.bar(correlation_with_Target_sorted.index,
                           correlation_with_Target_sorted.values)
            plt.xlabel('Columns')
            plt.ylabel(f'Correlation with {selected_column} ')
            plt.title(f'Correlation of {selected_column} with Other Columns')
            plt.xticks(rotation=90)
            plt.ylim(-1, 1)

            for i, bar in enumerate(bars):
                col_name = correlation_with_Target_sorted.index[i]
                col_correlation = bar.get_height()

                correlation_more_than_0_2 = [
                    col for col in correlation_with_Target_sorted.index
                    if correlation_with_Target_sorted[col] > 0.2
                ]
                correlation_less_than_minus_0_2 = [
                    col for col in correlation_with_Target_sorted.index
                    if -0.2 > correlation_with_Target_sorted[col]
                ]
                correlation_minus_0_2_to_0_2 = [
                    col for col in correlation_with_Target_sorted.index
                    if -0.2 <= correlation_with_Target_sorted[col] <= 0.2
                ]

            plt.grid(axis='y')
            png_path = 'static/images/correlation_graph.png'
            plt.savefig(png_path, bbox_inches='tight')
            drop_columns = correlation_minus_0_2_to_0_2
            col_cor = correlation_more_than_0_2 + correlation_less_than_minus_0_2
            drop_columns.append(selected_column)
            actual_columns = csv_data.columns.drop(drop_columns)
            # save selected column in a file
            with open("target_column.txt", "w") as file:
                file.write("\n".join(selected_column))
            # save actual columns in a file
            with open("highly_correlated_columns.txt", "w") as file:
                file.write("\n".join(actual_columns))
            if os.path.exists(png_path):
                return make_response({"message": "Correlation graph generated", "png_path": png_path}, 200)
            else:
                # 404 Not Found
                return make_response({"error": "Correlation graph not found"}, 404)


api.add_resource(CorrelationGraph, '/correlation-graph')

# Train the data


class TrainingModel(Resource):
    def get(self):
        with open("highly_correlated_columns.txt", "r") as file:
            actual_columns = file.read().splitlines()
        with open("target_column.txt", "r") as file:
            lines = file.readlines()
            selected_column = ''.join(line.strip() for line in lines)
        X = csv_data[actual_columns]
        y = csv_data[selected_column]
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Start measuring the training time
        start_time = time.time()

        # Train the Neural Network model
        MPL_model = MLPClassifier(max_iter=20)
        MPL_model.fit(X, y)
        with open('mpl_model.pkl', 'wb') as file:
            pkl.dump(MPL_model, file)

        # Calculate the training time
        end_time = time.time()
        training_time = end_time - start_time

        accuracy = cross_val_score(
            MPL_model, X, y, cv=cv, scoring='accuracy').mean()
        precision = cross_val_score(
            MPL_model, X, y, cv=cv, scoring='precision_macro').mean()

        return make_response({"message": "Model training completed successfully", 'accuracy': accuracy, 'precision': precision, 'training_time': training_time}, 200)


api.add_resource(TrainingModel, '/training')

# Predict


class PredictionFileUpload(Resource):
    def post(self):
        try:
            with open('mpl_model.pkl', 'rb') as file:
                model_pkl = pkl.load(file)
        except FileNotFoundError:
            error = 'Model file not found, Please train the model!'
            return make_response({'error': error}, 404)

        df = request.get_data()
        if not df:
            return make_response({"error": "No file was uploaded"}, 400)
        try:
            binary_io = io.BytesIO(df)
            data = pd.read_csv(binary_io) 
            
            with open("highly_correlated_columns.txt", "r") as file:
                actual_columns = file.read().splitlines()
            pred_X = data[actual_columns]

            y_pred = model_pkl.predict(pred_X)
            y_pred_series = pd.Series(y_pred)
            y_pred_string = y_pred_series.apply(
                lambda x: 'Dropout' if x == 0 else ('Graduate' if x == 1 else 'Enrolled'))
            y_pred_list = y_pred_string.tolist()

            return make_response({"message": "Prediction completed successfully", 'result': y_pred_list[0]}, 200)
        except pd.errors.ParserError as e:
            return make_response({"error": f"Error parsing CSV data: {str(e)}"}, 400)


api.add_resource(PredictionFileUpload, '/prediction/multiple-data')

class PredictionInput(Resource):
    def post(self):
        try:
            with open('mpl_model.pkl', 'rb') as file:
                model_pkl = pkl.load(file)
        except FileNotFoundError:
            error = 'Model file not found, Please train the model!'
            return make_response({'error': error}, 404)

        # df = request.files['file']
        # if not df:
        #     return make_response({"error": "No file was uploaded"}, 400)

        # Read the CSV file for prediction
        # data = pd.read_csv(df)

        with open("highly_correlated_columns.txt", "r") as file:
            actual_columns = file.read().splitlines()

        # Read input values from the request body as JSON
        input_data = request.get_json()

        if input_data is None:
            return make_response({"error": "No input data provided in the request body"}, 400)

        # Ensure that the input data keys match the columns in "highly_correlated_columns.txt"
        for column in actual_columns:
            if column not in input_data:
                return make_response({"error": f"Input data missing for column: {column}"}, 400)

        # Create a DataFrame with input values
        input_data_df = pd.DataFrame([input_data])

        pred_X = input_data_df[actual_columns]

        y_pred = model_pkl.predict(pred_X)
        y_pred_series = pd.Series(y_pred)
        y_pred_string = y_pred_series.apply(
            lambda x: 'Dropout' if x == 0 else ('Graduate' if x == 1 else 'Enrolled'))
        y_pred_list = y_pred_string.tolist()

        return make_response({"message": "Prediction completed successfully", 'result': y_pred_list[0]}, 200)


api.add_resource(PredictionInput, '/prediction/single-data')

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
