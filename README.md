# Hands on step 4: Logging and registering our model artifacts.

Now at last we want to finish the optuna study with retraining the winning configuration and logging it as a model that can be accessed through the mlflow model registry.
As usual check for TODO's in the files main.py and demo/loggers.py. It is recommended that you delete previous runs in mlflow (but do NOT delete the run!), and you can delete the study in the optuna-dashboard interface.

##Steps
1) Please implement the listed steps in the FinalLogger class under demo/loggers.py. It is based on the Mlflow logger class but has a more extensive model logging procedure.
We will first convert the model into a scripted model using torch.jit.script(model), which is handy for reducing the necessary source setups if the model is compatible. Log this scripted model using the pytorch flavour, 
and provide an input example to provide a mode signature. 

2) Next, update the OptunaRunner.finalize() function according to the listed instructions. Add a call to this in the run_project() function.

3) When all TODO's are completed, try running the optimization and check the results on mlflow. You may group by, or filter by the "status" tag set to "optimal" to find the optimal run. Note that it also points 
to a registered model, which points to a model artifact.

4) (Bonus) If you want you may now try serving the model through MLFLow by CLI. You need to be running inside the venv and ensure the system variable MLFLOW_TRACKING_URI=http://localhost:5000 (or whatever port your tracking server is running). Then we can run (windows cmd):

mlflow models serve -m "models:/MLP_circles/1" --no-conda -p 5001

(Which will launch a local server at port 5001)
If you manage to serve the model you may run ping.py that should generate a grid of points, send it to the server and plot the result.
