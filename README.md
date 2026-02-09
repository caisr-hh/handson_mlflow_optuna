# Hands on step 2: Pruning.

Now we attempt to speed up our search by pruning unpromising trials early.
As before check for TODO's in the files main.py and demo/loggers.py. It is recommended you delete the study in the optuna-dashboard interface, as otherwise it will be continued.

## Steps:
1)    Go to the OptunaStudyRunner and define the pruner in __init__(). Check the parameters n_startup_trials,n_warmup_steps in the loaded optuna config.
2)	Go to demo/loggers.py and look for the OptunaLogger. In the method that report loss for each training epoch,
      report this loss using trial.report(). We also need to know when to raise an exception and exit training, so we can poll trial.should_prune().
      This exception when thrown with context = "pruned" triggers other loggers to handle the interruption.
3)    Try running the HPO and verify that it is pruning trials. You may also check out the study in the optuna-dashboard.
