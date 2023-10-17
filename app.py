from flask import Flask, render_template,request,redirect,url_for
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle as pkl 

app = Flask(__name__)
message = 'message.html'
# home page
@app.route('/')
def home():
    return render_template('index.html')

# upload training data
@app.route('/upload-csv',methods=['GET','POST'])
def upload_csv():
    global csv_data
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            error = 'No file was uploaded'
            return render_template(message,error = error,path='upload-csv')
        if file:
            csv_data = pd.read_csv(file)
            label_mapping = {
                'Dropout': 0,
                'Graduate': 1,
                'Enrolled': 2
            }
            csv_data['Target'] = csv_data['Target'].map(label_mapping)
            return redirect(url_for("distribution_graph"))
    return render_template('upload_csv.html')

# Distribution Graph Page
@app.route("/distribution-graph", methods=["GET","POST"])
def distribution_graph():
    global csv_data
    if csv_data is not None:
        num_columns = len(csv_data.columns)
        num_rows = (num_columns + 4) // 5
        num_columns_last_row = num_columns % 5

        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=5, figsize=(15, 2 * num_rows))

        # Plot histograms for each column
        for i, col in enumerate(csv_data.columns):
            row_index = i // 5
            col_index = i % 5
            ax = axes[row_index, col_index]
            if i< num_columns:
                ax.hist(csv_data[col], bins=10)
                ax.set_title(col)
            else:
                ax.axis('off')

        if num_columns_last_row > 0:
            for j in range(num_columns_last_row, 5):
                axes[num_rows - 1, j].axis('off')
        plt.tight_layout()
        plt.savefig('static/images/distribution.png')
        return render_template('distribution.html',columns=list(csv_data.columns))
    return redirect(url_for("index.html"))

# correlation graph page
@app.route("/correlation-graph", methods=["GET","POST"])
def correlation_graph():
    global actual_columns, selected_column
    selected_column = request.form.get("selected_column")
    if not selected_column:
            error = 'No target column was selected'
            return render_template(message,error = error,path='upload-csv')
    
    if selected_column:
        correlation_with_Target = csv_data.corr()[selected_column]
        correlation_with_Target = correlation_with_Target.drop(selected_column)
        correlation_with_Target_sorted = correlation_with_Target.sort_values(ascending=False)

        # Plot the histogram
        plt.figure(figsize=(10, 6))
        bars = plt.bar(correlation_with_Target_sorted.index, correlation_with_Target_sorted.values)
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
                if  correlation_with_Target_sorted[col] > 0.2
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
        plt.savefig('static/images/correlation_graph.png', bbox_inches='tight')
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

    return render_template("correlation.html",selected_column=selected_column,col_no = correlation_minus_0_2_to_0_2,col_cor = col_cor)

@app.route("/training", methods=["GET","POST"])
def training_model():
    global accuracy,precision,selected_column,actual_columns
    X = csv_data[actual_columns]
    y = csv_data[selected_column]
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Train the Random Forest model
    MPL_model = MLPClassifier(max_iter=20)
    MPL_model.fit(X, y)
    with open('mpl_model.pkl', 'wb') as file:  
        pkl.dump(MPL_model, file)

    accuracy = cross_val_score(MPL_model, X, y, cv=cv, scoring='accuracy').mean()
    precision = cross_val_score(MPL_model, X, y, cv=cv, scoring='precision_macro').mean()
    return render_template('training.html',accuracy=accuracy,precision=precision)

@app.route("/prediction",methods=['GET','POST'])
def prediction_input():

    try:
        with open('mpl_model.pkl', 'rb') as file:
            model_pkl = pkl.load(file)
    except FileNotFoundError:
        error = 'Model file not found, Please train the model!'
        return render_template(message,error=error,path='upload-csv') 
    if request.method == 'POST':
        #request the file from the upload
        df = request.files['file']

        if not df:
            error = 'No file was uploaded'
            return render_template(message,error = error,path='prediction')

        # Read the CSV file for prediction
        data = pd.read_csv(df)
        
        with open("highly_correlated_columns.txt", "r") as file:
            actual_columns = file.read().splitlines()
        pred_X = data[actual_columns]

        y_pred = model_pkl.predict(pred_X)
        y_pred_series = pd.Series(y_pred)
        y_pred_string = y_pred_series.apply(lambda x: 'Dropout' if x == 0 else ('Graduate' if x == 1 else 'Enrolled'))
        y_pred_list = y_pred_string.tolist()
        print(y_pred_list)
        return render_template('result.html',result = y_pred_list[0])

    return render_template('prediction_form.html')

if __name__ == '__main__':
    app.run(debug=True,port=5000)