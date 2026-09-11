import optuna
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

cmap = cm.viridis

import numpy as np

n_trials = 100


n_samples = 1000

x = np.random.random(n_samples)
y = np.random.random(n_samples)

k,m = -1.25, 0.75
labels = (x*k + m > y)
dist = (x*k + m - y)




def objective(trial : optuna.Trial):
    _k = trial.suggest_float("k", -2,2)
    _m = trial.suggest_float("m", 0, 1)

    _dist = (x*_k + _m - y)

    _class =  (x*_k + _m > y).astype(float)
    accuracy = sum(abs(labels-_class))/n_samples
    return accuracy


study=optuna.create_study(study_name="test study",direction="minimize", storage="sqlite:///optuna.db", load_if_exists=True)


study.optimize(func = objective,n_trials = n_trials)


plt.scatter(x[labels], y[labels], marker = 'o', alpha=0.5)
plt.scatter(x[~labels], y[~labels], marker = 'v', alpha = 0.5)

for trial in study.trials:
    _k = trial.params["k"]
    _m = trial.params["m"]
    _x = np.array([0, 1])
    _y = np.array([_m, _k + _m])

    plt.plot(_x, _y, color = cmap(trial.number/n_trials), alpha = 0.5)

norm = mpl.colors.Normalize(vmin=0, vmax=n_trials)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

_x = np.array([0, 1])
_y = np.array([m, k + m])
plt.plot(_x, _y, 'r-')
plt.show()
