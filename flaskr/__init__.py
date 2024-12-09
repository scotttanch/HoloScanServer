import os
import logging
from datetime import datetime
from flask import Flask, flash, request, redirect
from flask import send_from_directory
from flaskr import create_resources

UPLOAD_FOLDER = 'Surveys'
ALLOWED_EXTENSIONS = {'csv', 'dzt'}
LOG_FOLDER = 'Logs'
time_frmt = "%y-%m-%d"
log_file = f"./flaskr/Logs/record_{datetime.now().strftime(time_frmt)}.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')


def allowed_file(filename):
	allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	return allowed


def create_index():
	storage_dir = "./flaskr/Surveys"

	manifest_dict = {}

	# get the survey folders, making sure that they are directories
	survey_folders = [sub_folder for sub_folder in os.listdir(storage_dir) if os.path.isdir(os.path.join(storage_dir, sub_folder))]

	# for each survey folder get their contents and add them to the dictionary if they are also directories
	for survey in survey_folders:
		survey_files = [sub_folder for sub_folder in os.listdir(os.path.join(storage_dir, survey)) if os.path.isdir(os.path.join(storage_dir, survey, sub_folder))]
		manifest_dict[survey] = survey_files

	# write out the manifest
	manifest_file = os.path.join(storage_dir, "_index.csv")
	with open(manifest_file, "w") as f:
		for key in manifest_dict:
			new_line = [key]
			new_line.extend(manifest_dict[key])
			str_line = ",".join(new_line)
			f.write(str_line + "\n")

	return


def create_app(test_config=None):
	app = Flask(__name__, instance_relative_config=True)
	app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
	app.config.from_mapping(
		SECRET_KEY='dev',
		DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite')
	)

	create_index()

	if test_config is None:
		app.config.from_pyfile('config.py', silent=True)
	else:
		app.config.from_mapping(test_config)

	try:
		os.makedirs(app.instance_path)
	except OSError:
		pass

	# Route/URL for HelloWorld Testing
	@app.route('/hello')
	def hello():
		return 'Hello, World'
	
	# Route/URL Pattern for main folder download. Applies only to _index.csv
	@app.route('/Surveys/<name>')
	def download_file(name):
		if name == '_index.csv':
			create_index()
		return send_from_directory(app.config["UPLOAD_FOLDER"], name)

	app.add_url_rule('/Surveys/<name>', endpoint="download_file", build_only=True)

	# Route/URL Pattern for obtaining a file specific file.
	@app.route('/Surveys/<survey>/<scan>/<file_name>')
	def new_download_file(survey, scan, file_name):
		return send_from_directory(app.config["UPLOAD_FOLDER"], f"{survey}/{scan}/{file_name}")
	
	app.add_url_rule('/Surveys/<survey>/<scan>/<file_name>', endpoint="download_file", build_only=True)

	# Most of this function is way more specific than it should be. But it does function
	@app.route('/', methods=['GET', 'POST'])
	def upload_file():
		if request.method == 'POST':
			if 'file' not in request.files:
				flash('No file part')
				return redirect(request.url)
			file = request.files['file']
			if file.filename == '':
				flash('No Selected file')
				return redirect(request.url)

			# Ensure that the file name is of the format survey_name/scan_name/file_name
			components = file.filename.split('/')
			if len(components) > 3:
				flash('Invalid file name')
				return redirect(request.url)

			# Need to now make sure that the folder we want exits
			# First see if the survey folder exists
			if not os.path.exists(f'./flaskr/Surveys/{components[0]}'):
				os.mkdir(f'./flaskr/Surveys/{components[0]}')
			if not os.path.exists(f'./flaskr/Surveys/{components[0]}/{components[1]}'):
				os.mkdir(f'./flaskr/Surveys/{components[0]}/{components[1]}')

			if file and allowed_file(file.filename):

				filename = file.filename

				file.save(f"./flaskr/Surveys/{filename}")

				folder = f'./flaskr/Surveys/{components[0]}/{components[1]}'
				ready_to_process = (any([content.startswith("FILE") for content in os.listdir(folder)]) and
								   any([content.startswith("PATH") for content in os.listdir(folder)]))
				if ready_to_process:
					print("Creating Resources")
					create_resources.create_resources(folder)
				else:
					print("Waiting for more data")

				return redirect(request.url)
		html_string = '''<!doctype html>
						<title>Upload new File</title>
						<h1>Upload new File</h1>
						<form method=post enctype=multipart/form-data>
							<input type=file name=file>
							<input type=submit value=Upload>
						</form>
					'''
		return html_string
	return app
