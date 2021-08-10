import flask
import os
from werkzeug.utils import secure_filename

from flask import url_for, send_from_directory

# Create the application
APP = flask.Flask(__name__)

# Set configuration parameters
APP.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static/uploads')
APP.add_url_rule(
    '/uploads/<name>', endpoint='display_file', build_only=True
)

@APP.route('/', methods = ['GET', 'POST'])
def index():
    if flask.request.method == 'POST':
        f = flask.request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(APP.config['UPLOAD_FOLDER'], filename))
        #return f'File {filename} Uploaded Successfully'
        return flask.redirect(url_for('display_file', name=filename))
    else:
        return flask.render_template('starter.html')

@APP.route('/uploads/<name>')
def display_file(name):
    #return 'File upload successful'
    return send_from_directory(APP.config['UPLOAD_FOLDER'], name)
    #return flask.render_template('image.html', image = os.path.join(APP.config['UPLOAD_FOLDER'], name), path = os.getcwd())

if __name__ == '__main__':
    APP.debug=True
    APP.run()