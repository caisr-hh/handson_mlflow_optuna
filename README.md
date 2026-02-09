# Hands on step  1: HPO with optuna.

Let us consider a very basic problem. We have an MLP for classification of 2d points and want to find an optimal width and depth for our network.

This pipeline relies on loggers collected under a shared pipelinelogger that forwards calls. We have a very simple local logger implemented as
our baseline. Look at the TODO's in the run_project function in main.py. Try running a single training session with the plotting on
before disabling it for future runs. 


As an Initial step try running the code as is 

Check for TODO's in the files main.py and demo/loggers.py. 

##Steps:
0) This pipeline relies on loggers collected under a shared pipelinelogger that forwards calls. We have a very simple local logger implemented as
 our baseline. Look at the TODO's in the run_project function in main.py. Try running a single training session with the plotting on
 before disabling it for future runs. 
1) Instead of running the pipeline directly, pass it to a OptunaStudyRunner.
2) Look at the initialization of OptunaStudyRunner and implement the necessary changes to set up the study. Note the config described by configs/optuna.yaml.
3) Go to the objective function (or method in this case) and suggest some integers with .suggest_int(name, low, high) for the width and depth configuration of the pipeline.
4) Setup the optimizer to call the right method.
5) Try running the HPO, if you have set up optuna-dashboard you should be able to see the study in the ui.