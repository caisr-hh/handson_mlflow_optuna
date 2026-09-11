# MLflow

To start a mlflow server, please start a terminal in this working directory and enter:

conda activate ecml_tutorial
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

This should start the trcking server locally at http://127.0.0.1:5000.

Once you have registered the model with mlflow_model_register.py you might attempt serving it in a terminal with:

## windows:
conda activate ecml_tutorial

set MLFLOW_TRACKING_URI http://127.0.0.1:5000

mlflow models serve -m "models:/registered_model/latest" --no-conda -p 5001

##Linux/mac:

conda activate ecml_tutorial

export MLFLOW_TRACKING_URI http://127.0.0.1:5000

mlflow models serve -m "models:/registered_model/latest" --no-conda -p 5001

If it hosts properly you can try sending and recieving payloads with ping.py


