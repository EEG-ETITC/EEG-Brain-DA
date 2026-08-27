import argparse
import os
import sys
from pathlib import Path

import mne
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from Models.Labram import MyLabram
from Utils.train import patient_stratify_split


DEFAULT_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
]


def load_labram_data(path_demographic="Code/d/data_patients.csv", path_files="Code/raw_tif/", hours=(36, 40, 44), channels=None):
    """Carga los datos EEG y las etiquetas demográficas para Labram."""
    if channels is None:
        channels = DEFAULT_CHANNELS

    df_demog = pd.read_csv(Path(ROOT_DIR, path_demographic), sep=",")
    df_demog["numeric_outcome"] = df_demog["Outcome"].map({"Good": 1, "Poor": 0})
    df_demog.reset_index(drop=True, inplace=True)

    raw_root = Path(ROOT_DIR, path_files)
    hour_list = [x for x in os.listdir(raw_root) if int(x.split("h")[1]) in hours]

    raws = []
    patients = []
    for hour_dir in hour_list:
        file_list = os.listdir(raw_root / hour_dir)
        for file_name in file_list:
            raw_path = raw_root / hour_dir / file_name
            raws.append(mne.io.read_raw_fif(raw_path, preload=True))
            patients.append(file_name.split(".")[0][-4:])

    X = []
    scaler = StandardScaler()
    demograp_data = pd.DataFrame(columns=df_demog.columns)
    for raw, patient in zip(raws, patients):
        patient_row = df_demog[df_demog["Id_Patient"] == int(patient)]
        if patient_row.empty:
            continue
        demograp_data = pd.concat([demograp_data, patient_row], ignore_index=True)

        eeg_data = raw.get_data(picks=channels)
        eeg_data = scaler.fit_transform(eeg_data.T).T
        X.append(eeg_data.astype(np.float32))

    X = np.stack(X, axis=0)
    y = demograp_data["numeric_outcome"].astype(np.int64).to_numpy()

    return X, y, demograp_data


def build_objective(X_train, y_train, X_val, y_val, n_times, n_chans, n_outputs, channels, sfreq=128):
    """Construye la función objetivo para Optuna."""

    def objective(trial):
        params = {
            "patch_size": trial.suggest_categorical("patch_size", [128, 256]),
            "embed_dim": trial.suggest_categorical("embed_dim", [64, 128, 256]),
            "num_layers": trial.suggest_categorical("num_layers", [4, 8, 12]),
            "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
            "mlp_ratio": trial.suggest_categorical("mlp_ratio", [2, 4, 6]),
            "drop_prob": trial.suggest_float("drop_prob", 0.0, 0.3),
            "attn_drop_prob": trial.suggest_float("attn_drop_prob", 0.0, 0.3),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "n_epochs": trial.suggest_categorical("n_epochs", [5, 8, 10]),
        }

        activation_name = trial.suggest_categorical("activation", ["gelu", "relu", "silu"])
        activation_map = {"gelu": torch.nn.GELU, "relu": torch.nn.ReLU, "silu": torch.nn.SiLU}
        activation = activation_map[activation_name]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(2024)

        model = MyLabram(
            n_times=n_times,
            n_chans=n_chans,
            n_outputs=n_outputs,
            ch_names=channels,
            sfreq=sfreq,
            patch_size=params["patch_size"],
            embed_dim=params["embed_dim"],
            num_layers=params["num_layers"],
            num_heads=params["num_heads"],
            mlp_ratio=params["mlp_ratio"],
            drop_prob=params["drop_prob"],
            attn_drop_prob=params["attn_drop_prob"],
            activation=activation,
        )
        model.to(device)

        clf = torch.nn.Module()  # placeholder to satisfy linter; overwritten below
        from braindecode import EEGClassifier

        clf = EEGClassifier(
            module=model,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            train_split=None,
            batch_size=params["batch_size"],
            optimizer__lr=params["lr"],
            optimizer__weight_decay=params["weight_decay"],
            max_epochs=params["n_epochs"],
            device=device,
            classes=np.unique(y_train),
            verbose=0,
        )

        clf.fit(X_train, y_train)
        val_prob = clf.predict_proba(X_val)
        if val_prob.shape[1] == 1:
            val_score = val_prob[:, 0]
        else:
            val_score = val_prob[:, 1]
        return roc_auc_score(y_val, val_score)

    return objective


def run_optuna_search(
    X,
    y,
    demograp_data,
    train_size=0.8,
    n_trials=15,
    n_jobs=1,
    study_name="labram_optuna",
    storage=None,
    seed=2024,
    channels=None,
    sfreq=128,
):
    """Ejecuta la búsqueda de hiperparámetros con Optuna."""
    if channels is None:
        channels = DEFAULT_CHANNELS

    patients_train, patients_val = patient_stratify_split(demograp_data, train_size=train_size)
    X_train, y_train = X[patients_train], y[patients_train]
    X_val, y_val = X[patients_val], y[patients_val]

    n_times = X_train.shape[-1]
    n_chans = X_train.shape[1]
    n_outputs = len(np.unique(y_train))

    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage, load_if_exists=True)
    objective = build_objective(
        X_train,
        y_train,
        X_val,
        y_val,
        n_times=n_times,
        n_chans=n_chans,
        n_outputs=n_outputs,
        channels=channels,
        sfreq=sfreq,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params["value"] = best_trial.value
    return study, best_params


def main():
    parser = argparse.ArgumentParser(description="Búsqueda de hiperparámetros de Labram con Optuna")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--study-name", type=str, default="labram_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--hours", type=int, nargs="+", default=[36, 40, 44])
    parser.add_argument("--sfreq", type=int, default=128)
    args = parser.parse_args()

    X, y, demograp_data = load_labram_data(hours=tuple(args.hours))
    study, best_params = run_optuna_search(
        X,
        y,
        demograp_data,
        train_size=args.train_size,
        n_trials=args.trials,
        n_jobs=args.n_jobs,
        study_name=args.study_name,
        storage=args.storage,
        sfreq=args.sfreq,
    )

    print("\nMejor trial:")
    print(best_params)
    print("\nMejores parámetros:")
    print(study.best_params)


if __name__ == "__main__":
    main()
