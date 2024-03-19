import optuna
from optuna_dashboard import run_server

def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_float("y", -100, 100)
    return x**2 + y

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(storage=storage, direction="minimize")
study.optimize(objective, n_trials=1)

run_server(storage)