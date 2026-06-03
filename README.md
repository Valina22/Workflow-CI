# Workflow-CI

Automated Machine Learning retraining workflow using:

* MLflow Projects
* GitHub Actions
* DagsHub MLflow Tracking
* Scikit-Learn

## Project Structure

* MLProject

  * modelling.py
  * conda.yaml
  * MLproject
  * dataset_preprocessing

## Workflow

The workflow automatically retrains the model whenever a push is made to the main branch.

## Experiment Tracking

MLflow experiments are tracked through DagsHub.

## Author

Valina Puspita Sari
