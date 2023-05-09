from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the machine learning model
ml_mode_path = 'D:\\Assignment\\05 machine learning\\project_Nifty50\\lr.pkl'

# model = pickle.load(open(ml_mode_path, 'rb'))
model = joblib.load(ml_mode_path)

@app.route("/", methods=["GET", "POST"])
def home():
    app_data = pd.read_csv("D:\\Assignment\\05 machine learning\\project_Nifty50\\stock_name_binary.csv")
    stock_name_list = list(app_data["Stock Name"])

    if request.method == "POST":
        # Get the input values from(user) the form
        stock_name = request.form['a']
        total_quantity = float(request.form['b'])
        opening_stock = float(request.form['c'])

        if stock_name == "":
            stock_name = "ADANIENT"

        row = list(app_data["Stock Name"]).index(stock_name)
        l1 = list(app_data.iloc[row,1:])
        l1.insert(0, opening_stock)
        l1.insert(1, total_quantity)
        test = np.array(l1)
        
        # # Make a prediction using the machine learning model
        closing_stock_pred = round(model.predict(test.reshape(1, 52))[0], 2)
        PandL_pred = round(opening_stock - closing_stock_pred, 2)

        # Render the home page with the predicted output values
        return render_template('index.html', closing_stock = closing_stock_pred, PandL = PandL_pred, stock_name_list = stock_name_list)
    
    # Render the home page without any predicted output values
    return render_template("index.html", stock_name_list = stock_name_list)

if __name__ == "__main__":
    app.run(debug=True)
