from flask import Flask, request, make_response
from flask_restful import Resource, Api
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle as pkl
import os

app = Flask(__name__)
api = Api(app)

# home page


class Home(Resource):
    def get(self):
        return {"message": "Welcome to the API"}


api.add_resource(Home, '/')

# upload training data


class UploadCSV(Resource):
    def post(self):
        global csv_data
        file = request.files['file']
        if not file:
            return make_response({"error": "No file was uploaded"}, 400)
        csv_data = pd.read_csv(file)
        label_mapping = {'Dropout': 0, 'Graduate': 1, 'Enrolled': 2}
        csv_data['Target'] = csv_data['Target'].map(label_mapping)
        return {"message": "CSV data uploaded successfully"}


api.add_resource(UploadCSV, '/upload-csv')

# Distribution Graph Page


class DistributionGraph(Resource):
    def get(self):
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
            return make_response({"error": "No CSV data available"}, 403)


api.add_resource(DistributionGraph, '/distribution-graph')

# correlation graph page


class CorrelationGraph(Resource):
    def get(self):
        global actual_columns, selected_column
        selected_column = request.form.get("selected_column")
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


class PredictionInput(Resource):
    def post(self):
        try:
            with open('mpl_model.pkl', 'rb') as file:
                model_pkl = pkl.load(file)
        except FileNotFoundError:
            error = 'Model file not found, Please train the model!'
            return make_response({'error': error}, 404)

        df = request.files['file']
        if not df:
            return make_response({"error": "No file was uploaded"}, 400)

        # Read the CSV file for prediction
        data = pd.read_csv(df)

        with open("highly_correlated_columns.txt", "r") as file:
            actual_columns = file.read().splitlines()
        pred_X = data[actual_columns]

        y_pred = model_pkl.predict(pred_X)
        y_pred_series = pd.Series(y_pred)
        y_pred_string = y_pred_series.apply(
            lambda x: 'Dropout' if x == 0 else ('Graduate' if x == 1 else 'Enrolled'))
        y_pred_list = y_pred_string.tolist()

        return make_response({"message": "Prediction completed successfully", 'result': y_pred_list[0]}, 200)


api.add_resource(PredictionInput, '/prediction')

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
