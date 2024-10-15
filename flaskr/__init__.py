import os
import logging
from datetime import datetime
from flask import Flask
from flask import send_from_directory

UPLOAD_FOLDER = 'Surveys'
LOG_FOLDER = 'Logs'
time_frmt = "%y-%m-%d"
log_file = f"./flaskr/Logs/record_{datetime.now().strftime(time_frmt)}.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format = '%(message)s')


def create_manifest():
	storage_dir = "./flaskr/Surveys"

	manifest_dict = {}

	# get the survey folders, making sure that they are directories
	survey_folders = [sub_folder for sub_folder in os.listdir(storage_dir)
					  if os.path.isdir(os.path.join(storage_dir, sub_folder))]

	# for each survey folder get their contents and add them to the dictionary if they are also directories
	for survey in survey_folders:
		survey_files = [sub_folder for sub_folder in os.listdir(os.path.join(storage_dir, survey))
						if os.path.isdir(os.path.join(storage_dir, survey, sub_folder))]
		manifest_dict[survey] = survey_files

	# write out the manifest
	manifest_file = os.path.join(storage_dir, "_manifest.csv")
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

	create_manifest()

	if test_config is None:
		app.config.from_pyfile('config.py', silent=True)
	else:
		app.config.from_mapping(test_config)

	try:
		os.makedirs(app.instance_path)
	except OSError:
		pass

	@app.route('/hello')
	def hello():
		return 'Hello, World'
	
	# this route is basically just for downloading the manifest but im leaving it as is in case I need it again later
	@app.route('/Surveys/<name>')
	def download_file(name):
		if name == '_manifest.csv':
			#print("updating manifest")
			create_manifest()
		return send_from_directory(app.config["UPLOAD_FOLDER"], name)

	app.add_url_rule('/Surveys/<name>',
					 endpoint="download_file",
					 build_only=True)

	# this route is for downloading things from the file tree, it has to be structured like this, or it doesn't know how to deal with the resulting url
	# I think it could be handled more elegantly, but we aren't in the business of elegance
	@app.route('/Surveys/<survey>/<scan>/<file_name>')
	def new_download_file(survey, scan, file_name):
		return send_from_directory(app.config["UPLOAD_FOLDER"], f"{survey}/{scan}/{file_name}")
	
	app.add_url_rule('/Surveys/<survey>/<scan>/<file_name>',
					 endpoint="download_file",
					 build_only=True)
	
	return app
