# Hands on step 4: Connecting to mlflow, starting runs and logging.

Now we look at mlflow integration, so ensure that your tracking server is running according to the configuration in config/mlflow.yaml.
As usual check for TODO's in the files main.py and demo/loggers.py. It is recommended you delete the study in the optuna-dashboard interface, as otherwise it will be continued.

##Steps:
1)	Go to the optunarunner.objective function and call mlflowdriver. This defaults to using the MLFlowLogger class.
2)	Go to the half finished mlflowdriver function and set the tracking and experiment. 
3)	Let us now go to the loggers and finish implementing the MLFlowLogger class. As the calls to the logger we are interested in occur in the context of an active run, the runid is handled automatically. We will log epoch training loss, test loss and accuracy, model parameters and a figure.
      Some useful functions are:
      mlflow.
            log_metrics(dict,step)
            log_params(dict)
            set_tag(key, value)
            log_figure(fig,artifact_file)
            log_text(text,artifact_file)
      
4)	Try running the code with your tracker server running. Follow the address to make sure your runs are logged and look at their metrics, parameters, tags and artifacts.
