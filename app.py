from flask import Flask
from flask_restful import Api
import swagger_config as sw
from app.routes.home import Home
from app.routes.upload_csv import UploadCSV
from app.routes.distribution_graph import DistributionGraph
from app.routes.correlation_graph import CorrelationGraph
from app.routes.training_model import TrainingModel
from app.routes.predict_multiple import PredictionFileUpload
from app.routes.predict_single import PredictionInput

app = Flask(__name__)
api = Api(app)


# swagger config
app.register_blueprint(sw.SWAGGER_BLUEPRINT, specs_url=sw.SWAGGER_URL)

# home page
api.add_resource(Home, '/')

# upload training data
api.add_resource(UploadCSV, '/upload-csv')

# Distribution Graph Page
api.add_resource(DistributionGraph, '/distribution-graph')

# correlation graph page
api.add_resource(CorrelationGraph,
                 '/correlation-graph/<string:selected_column>')

# Train the data
api.add_resource(TrainingModel, '/training')

## Predict ##
# Predict multiple records
api.add_resource(PredictionFileUpload, '/prediction/multiple-data')

# Predict single record
api.add_resource(PredictionInput, '/prediction/single-data')


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
