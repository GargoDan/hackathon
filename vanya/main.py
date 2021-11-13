import os
from werkzeug.utils import secure_filename
from flask import Flask,flash,request,redirect,send_file,render_template
#from flask import json
from time import sleep


UPLOAD_FOLDER = '/home/ivan/files/pyfiles/hack_serv/data/'

#app = Flask(__name__)
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Upload API
@app.route('/medicine', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('no filename')
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("saved file successfully")
            sleep(5)
            return redirect('/download')

    return render_template('upload_file.html')


# Download API
@app.route("/download", methods = ['GET'])
def download_file():
    return render_template('download.html')

# @app.route('/dict')
# def summary():
#     file_path = UPLOAD_FOLDER + "dict.json"
#     return send_file(file_path, as_attachment=True, attachment_filename='')

@app.route('/return-files')
def return_files_tut():
    file_path = UPLOAD_FOLDER + "predictions.csv"
    return send_file(file_path, as_attachment=True, attachment_filename='')

# @app.route('/return-files/<region>')
# def return_files_tut(region):
#     file_path = UPLOAD_FOLDER + "predictions.csv"
#     return send_file(file_path, as_attachment=True, attachment_filename='')

if __name__ == "__main__":
    app.run(host='0.0.0.0')