from flask import Flask, render_template
from flask import request
from flask import redirect, url_for
import os
import pickle
import numpy as np
import pandas as pd
import scipy
import sklearn
import skimage
import skimage.color
import skimage.transform
import skimage.feature
import skimage.io
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH,'static/models/')

## -------------------- Load Models -------------------
model_lgr_path = os.path.join(MODEL_PATH,'logical_Reg_fishSpecies.pickle')
model_lgr = pickle.load(open(model_lgr_path,'rb'))


@app.errorhandler(404)
def error404(error):
    message = "ERROR 404 OCCURED. Page Not Found. Please go the home page and try again"
    return render_template("error.html",message=message) # page not found

@app.errorhandler(405)
def error405(error):
    message = 'Error 405, Method Not Found'
    return render_template("error.html",message=message)

@app.errorhandler(500)
def error500(error):
    message='INTERNAL ERROR 500, Error occurs in the program'
    return render_template("error.html",message=message)


@app.route('/',methods=['GET','POST'])
def index():
    print('Test',request.method)
    if request.method == "POST":
        pred_val= list(map(float,[x for x in request.form.values()]))
        print(pred_val)
        
        final_features = [np.array(pred_val)]
        print(final_features)
        results = model_lgr.predict(final_features) 
        print(results[0])        
        fishname=results[0]                 
        return render_template('upload.html',success=True,result_msg="The fish belong to species {}".format(fishname))
    else:
        return render_template('upload.html',success=False,extension=False)


@app.route('/about/')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=False) 