from flask import request, make_response
from flask_restful import Resource
import matplotlib.pyplot as plt
import os
from app.routes.distribution_graph import DistributionGraph

class CorrelationGraph(Resource):
    def calculate_correlation(self, selected_column):
        csv_data_instance = DistributionGraph()
        csv_data = csv_data_instance.fetch_csv_data()
        # Perform correlation calculation
        correlation_with_Target = csv_data.corr()[selected_column]
        correlation_with_Target = correlation_with_Target.drop(selected_column)
        return correlation_with_Target.sort_values(ascending=False)

    def generate_correlation_graph(self, correlation_data, selected_column):
        # Plot the histogram
        plt.figure(figsize=(10, 6))
        bars = plt.bar(correlation_data.index, correlation_data.values)
        plt.xlabel('Columns')
        plt.ylabel(f'Correlation with {selected_column} ')
        plt.title(
            f'Correlation of {selected_column} with Other Columns')
        plt.xticks(rotation=90)
        plt.ylim(-1, 1)
        plt.grid(axis='y')

    def save_correlation_image(self):
        # Ensure the directories exist
        if not os.path.exists("static/images"):
            os.makedirs("static/images")
        # Save the image with an absolute path
        png_path = os.path.abspath('static/images/correlation_graph.png')
        plt.savefig(png_path, bbox_inches='tight')
        return png_path

    def find_highly_correlated_columns(self, correlation_data, lower_threshold=-0.2, upper_threshold=0.2):
        return [col for col in correlation_data.index if not (lower_threshold <= correlation_data[col] <= upper_threshold)]

    def open_file(self, columns, filename, mode):
        with open(filename, mode) as file:
            file.write("\n".join(columns))

    def get(self, selected_column):
        try:
            if not selected_column:
                return make_response({"error": 'No target column was selected'}, 400)
            else:

                # Calculate correlation
                correlation_data = self.calculate_correlation(selected_column)
                # Generate correlation graph
                self.generate_correlation_graph(
                    correlation_data, selected_column)
                # Save the image
                png_path = self.save_correlation_image()
                # find highly correlated columns
                highly_correlated = self.find_highly_correlated_columns(
                    correlation_data)
                # save highly correlated columns to a text file
                self.open_file(highly_correlated,
                               "highly_correlated_columns.txt", "w")
                # save target column to a text file
                self.open_file(selected_column, "target_column.txt", "w")

            if os.path.exists(png_path):
                return make_response({"message": "Correlation graph generated", "png_path": png_path}, 200)
            else:
                # 404 Not Found
                return make_response({"error": "Correlation graph not found"}, 404)
        except Exception as e:
            return make_response({"error": f"An error occurred: {str(e)}"}, 500)
