import pandas as pd
from flask import Flask, render_template, request
app = Flask(__name__)
import pickle
import joblib
from sqlalchemy import create_engine

# Load the Model
# Load the preprocessed Data.
model = pickle.load(open("mpg.pkl", 'rb'))
ct = joblib.load('column1')

"""engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",# user
                               pw = "....", # passwrd
                               db = "ds")) #database
"""
# Flask api points to DIR /templates
@app.route('/')
def home():
    return render_template("index.html")


# Using Post,
# Inputs : 1 csv file
# Output : A HTML webpage with the predicted output.
@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data = pd.read_csv(f)

        data1 = data.to_numpy()
        
        y2 = pd.DataFrame(model.predict(ct.transform(data1)))
        
        data['MPG'] = y2
        #data.to_sql('result', con = engine, if_exists = 'append', chunksize = 1000)
        #data.to_sql('cars', con = engine, if_exists = 'replace', chunksize = 1000,schema='ds',index=False)
        return render_template("data.html", Z = "Your results are here", Y = data.to_html())

if __name__ == '__main__':

    app.run(debug = True)
