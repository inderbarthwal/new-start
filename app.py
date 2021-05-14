
from flask import Flask, redirect, url_for, request, jsonify, render_template
import os
import glob
import ResNet_arch as arch

from werkzeug.utils import secure_filename
from test import predict, test1

 

app = Flask(__name__)


@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/Upload',methods=['GET','POST'])
def uploads():
    if request.method == 'POST':

        f = request.files['file']           

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'LR', secure_filename(f.filename))
        f.save(file_path)

        x = test1()

        return render_template("base.html", name = x + ".png")
    return None


if __name__ == '__main__':
    app.run(debug=True)




