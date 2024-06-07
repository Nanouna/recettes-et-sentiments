
GS_DIR=gs://datascience-mlops/taxi-fare-ny
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y recettes-et-sentiments || :
	@pip install -e .

run_api:
	gsutil cp gs://recettes-et-sentiments/r_et_s.model.zip /prod/
	gsutil cp gs://recettes-et-sentiments/r_et_s_2.model.zip /prod/
	gsutil cp gs://recettes-et-sentiments/RAW_recipes.csv.zip /prod/

	uvicorn taxifare.api.fast:app --reload

##################### TESTS #####################

################### DATA SOURCES ACTIONS ################
