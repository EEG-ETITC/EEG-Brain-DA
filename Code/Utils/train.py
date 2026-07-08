# Módulo de entrenamiento y evaluación para modelos de clasificación sobre señales EEG.
# Aquí se preparan los datos, se entrenan modelos con validación cruzada y se reportan métricas.

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from numpy import array, bincount
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score,  recall_score, precision_score, f1_score, roc_curve, auc, classification_report
from Utils.download import paching
from sklearn.preprocessing import StandardScaler
import pandas as pd
from numpy import random, concatenate
import torch
import torch.nn as nn
from torch.functional import F
import torch.optim as optim
from seaborn import heatmap
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    """Contenedor simple para exponer los datos EEG como un dataset de PyTorch."""

    def __init__(self, X, y):
        # Guarda las características y etiquetas del conjunto.
        self.X = X
        self.y = y

    def __len__(self):
        # Devuelve la cantidad total de ejemplos disponibles.
        return len(self.X)

    def __getitem__(self, idx):
        # Retorna un ejemplo específico como par (características, etiqueta).
        return self.X[idx], self.y[idx]


def evaluar_modelo_clasificador(modelo, grid_params: dict, X: array, y: array, k: int, tipo_kfold="kfold"):
    """
    Entrena un modelo con GridSearchCV y devuelve un resumen con el mejor puntaje.
    Permite elegir entre validación cruzada estándar o estratificada.
    """
    # Define el tipo de partición que se usará para la validación cruzada.
    if tipo_kfold == "kfold":
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
    else:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    grid = GridSearchCV(modelo, grid_params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)
    grid.fit(X, y)

    return [{
        "Modelo": type(modelo).__name__,
        "KFold": tipo_kfold,
        "Mejor Puntaje": grid.best_score_,
        "Mejores Parámetros": grid.best_params_,
        "k": k
    }, grid]
    

def data_train(raw_list: list, df_demog: array, wind_size: int, over_lap: int, patients: list, channels: list, scaler: bool = True):
    """
    Prepara los datos EEG para entrenamiento a partir de una lista de señales crudas.
    Convierte cada señal en patches y une la información demográfica correspondiente.
    """
    data = []
    if scaler:
        # Instancia el escalador para normalizar cada canal.
        Scaler = StandardScaler()

    # Acumula la información demográfica de todos los pacientes procesados.
    demograp_data = pd.DataFrame(columns=df_demog.columns)
    for k, raw in enumerate(raw_list):
        patient = patients[k]
        demograp_data = pd.concat([demograp_data, df_demog[df_demog["Id_Patient"] == int(patient)]])
        df_aux = pd.DataFrame(raw.get_data().T, columns=raw.ch_names)

        df_aux = df_aux[channels]
        if scaler:
            df_aux = Scaler.fit_transform(df_aux).T
        else:
            df_aux = df_aux.T

        # Genera los patches de la señal para alimentar al modelo.
        patches, pos = paching(df_aux, window_size=wind_size, overlapping_size=over_lap)
        data.append(patches)

    return data, demograp_data


def cv_patients(patients: list, n_split=5, shuffle=True, random_state=0) -> list:
    """Genera particiones de pacientes para validación cruzada por folds."""
    random.seed(random_state)
    n_split = 5

    # Determina el orden de los pacientes para crear los folds.
    if shuffle:
        index = random.permutation(pd.unique(patients))
    else:
        index = pd.unique(patients)

    # Calcula el tamaño de cada partición.
    len_partition = len(index) // n_split
    indexes = [index[i * len_partition:(i + 1) * len_partition] for i in range(n_split - 1)]
    indexes.append(index[(n_split - 1) * len_partition:])
    Idx = []

    # Crea los folds dejando uno como conjunto de prueba y el resto como entrenamiento.
    for i in range(n_split):
        idx_train = [indexes[j] for j in range(n_split) if j != i]
        idx_train = concatenate(idx_train).tolist()
        idx_test = indexes[i].tolist()
        Idx.append((idx_train, idx_test))

    return Idx


def patient_stratify_split(demograp_data, train_size=0.8, shuffle=True):
    """Divide los pacientes manteniendo la proporción de clases en los conjuntos."""
    # Obtiene una etiqueta por paciente para conservar el equilibrio de clases.
    Y = demograp_data.reset_index().drop_duplicates(subset=["Id_Patient"])["numeric_outcome"]
    etiquetas = Y.value_counts(normalize=True).index.tolist()

    indx_train = []
    indx_val = []

    # Para cada clase, toma una muestra de entrenamiento y el resto como validación.
    for l in etiquetas:
        y = pd.unique(demograp_data[demograp_data["numeric_outcome"] == l].index.tolist())
        if shuffle:
            y = random.permutation(y)
        indx_train.append(random.choice(y, size=int(train_size * len(y)), replace=False).tolist())
        indx_val.append(list(set(y) - set(indx_train[-1])))

    patients_train = [item for sublist in indx_train for item in sublist]
    patients_val = [item for sublist in indx_val for item in sublist]

    return patients_train, patients_val
############################################################


def train(model, X_train, y_train, patients, n_splits, demograp_train_data,
          batch_size: int, epochs: int, device, lr=1e-5):
    """
    Entrena un modelo con validación cruzada por pacientes usando DataLoader de PyTorch.
    Devuelve el modelo entrenado y las curvas de precisión y pérdida acumuladas.
    """
    accuracy_test = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    Acc_scores = []
    Loss_scores = []
    Cumulate_scores=[]
    Cumulate_loss=[]
    
    for fold, (train_idx, test_idx) in enumerate(cv_patients(patients, n_split=n_splits)):
        ind_train=demograp_train_data.loc[train_idx]["indices"].values
        ind_test=demograp_train_data.loc[test_idx]["indices"].values
        # ---------- DATASET POR FOLD ----------
        train_dataset = EEGDataset(X_train[ind_train], y_train[ind_train])
        test_dataset  = EEGDataset(X_train[ind_test], y_train[ind_test])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        model = model.to(device)
        class_counts = bincount(y_train)
        weights = 1.0 / class_counts
        weights = torch.tensor(weights, dtype=torch.float).to(device)

        criterion = nn.CrossEntropyLoss(weight=weights)
        # criterion=nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=max(1, total_steps)
        )

        # ---------- TRAIN LOOP ----------
        for epoch in range(epochs):

            model.train()

            for X_batch, y_batch in train_loader:

                X_batch = X_batch.float().to(device, non_blocking=True)
                y_batch = y_batch.long().to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda",
                                    dtype=torch.float16,
                                    enabled=(device=="cuda")):

                    outputs = model(X_batch)
                    outputs=F.softmax(outputs)
                    loss = criterion(outputs, y_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # ---------- evaluación entrenamiento ----------
            model.eval()
            
            with torch.no_grad():
                outputs_test = model(X_train[ind_test].float().to(device, non_blocking=True))
                Loss_scores.append(loss.item())
                Cumulate_loss.append(sum(Loss_scores)/len(Loss_scores))
                pred = outputs_test.argmax(dim=1)
                acc_train = (pred == y_train[ind_test].float().to(device, non_blocking=True)).float().mean().item()
                Acc_scores.append(acc_train)
                Cumulate_scores.append(sum(Acc_scores)/len(Acc_scores))

                if epoch % 5 == 0:
                    print(f"[Fold {fold+1}] Época {epoch}: pérdida={sum(Loss_scores)/len(Loss_scores):.3f}, acc_train={sum(Acc_scores)/len(Acc_scores):.3f}")

        # ---------- VALIDACIÓN ----------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:

                X_batch = X_batch.float().to(device, non_blocking=True)
                y_batch = y_batch.long().to(device, non_blocking=True)

                with torch.autocast(device_type="cuda",
                                    dtype=torch.float16,
                                    enabled=(device=="cuda")):

                    logits = model(X_batch)
                    logits=F.softmax(logits)

                preds = torch.argmax(logits, dim=1)

                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        acc = correct / total
        accuracy_test += acc

        print(f"[Fold {fold+1}] Precisión validación: {acc:.3f}")

        torch.cuda.empty_cache()

    print("Average kfold accuracy: ", accuracy_test / n_splits)

    return model, Cumulate_scores, Cumulate_loss







###################################################################
def eval(model, X_test, y_test, model_name: str, S_labels: list, Cumulate_scores: list = None, Cumulate_loss: list = None,
         path_confusion=None, path_f1score=None, save_fig=True, normalize_conf_matr="True"):
    """
    Evalúa un modelo entrenado sobre un conjunto de prueba y genera métricas y gráficos.
    Incluye matriz de confusión, métricas clásicas y curvas ROC para problemas binarios.
    """
    data_test = EEGDataset(X_test, y_test)

    test_loader=DataLoader(data_test, batch_size=16, shuffle=True)
    model.cpu()
    torch.cuda.amp.autocast("cuda")

    model.eval()

    with torch.no_grad():
        y_score=[]
        y_pred=[]
        y=[]
        for X_batch, y_batch in test_loader:

            X_batch = X_batch.float().cpu()
            y_batch = y_batch.float().cpu()

            logits = model(X_batch)

            y_pred.append(torch.argmax(logits, dim=1))
            y_score.append(torch.softmax(logits, dim=1))
            y.append(y_batch)
            
            del X_batch

    y_score=torch.cat(y_score)
    y_pred=torch.cat(y_pred)
    y_test_cpu=torch.cat(y)


    if not Cumulate_loss is None and not Cumulate_scores is None:
        # -------- curvas entrenamiento --------
        plt.figure()
        plt.plot(Cumulate_scores)
        plt.title("Accuracy performance")
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.show()

        plt.figure()
        plt.plot(Cumulate_loss)
        plt.title("Loss performance")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.show()

    # -------- métricas --------
    m_conf = confusion_matrix(y_test_cpu, y_pred, normalize=normalize_conf_matr)

    accuracy_test = accuracy_score(y_test_cpu, y_pred)
    recall = recall_score(y_test_cpu, y_pred, average="weighted")
    precision = precision_score(y_test_cpu, y_pred, average="weighted")
    f1 = f1_score(y_test_cpu, y_pred, average="weighted")
    cl_report=classification_report(y_test_cpu, y_pred,)
    
    plt.figure()
    heatmap(m_conf, annot=True, fmt=".2g",
            xticklabels=S_labels,
            yticklabels=S_labels)

    
    plt.title(f"{model_name}")
    if save_fig:
        plt.savefig(path_confusion, dpi=150)
    plt.show()

    print("Exactitud:", accuracy_test)
    print("Exhaustividad:", recall)
    print("Precisión:", precision)
    print("f1_score:", f1)

    # -------- ROC binaria --------
    if len(torch.unique(y_test_cpu)) == 2:

        for i in [0,1]:
            fpr, tpr, thresholds = roc_curve(y_test_cpu, y_score[:,i])
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(7,5))
            plt.plot(fpr, tpr, lw=2,
                        label=f'ROC (AUC = {roc_auc:.2f})')
            plt.plot([0,1], [0,1], linestyle='--')

            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title(f'Curva ROC {S_labels[i]}')
            plt.legend(loc='lower right')
            plt.grid(True)
            if save_fig:
                plt.savefig(path_f1score+S_labels[i]+".jpg", dpi=150)
            plt.show()
        return (m_conf, accuracy_test, recall, precision, f1, cl_report, roc_auc, fpr, tpr, thresholds)
    
    return (m_conf, accuracy_test, recall, precision, f1, cl_report)