from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from numpy import array
from Utils.download import paching
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup




def evaluar_modelo_clasificador(modelo, grid_params: dict, X: array, y: array, k: int, tipo_kfold="kfold" ):
    """
    Entrena un modelo con GridSearchCV y devuelve los resultados en DataFrame.
    tipo_kfold puede ser 'kfold' o 'stratified'.
    """
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
    

def data_train(raw_list: list, df_demog: array, wind_size: int, over_lap: int, patients:list, channels:list,  scaler: bool= True):
    data=[]
    if scaler:
        Scaler=StandardScaler()
    demograp_data=pd.DataFrame(columns=df_demog.columns)
    for k, raw in enumerate(raw_list):
        patient=patients[k]
        demograp_data=pd.concat([demograp_data,df_demog[df_demog["Id_Patient"]==int(patient)]])
        df_aux=pd.DataFrame(raw.get_data().T, columns=raw.ch_names)

        df_aux=df_aux[channels]
        if scaler:
            df_aux=Scaler.fit_transform(df_aux).T
        else:
            df_aux=df_aux.T
        
        patches, pos =paching(df_aux,
                            window_size=wind_size,
                            overlapping_size=over_lap)
        data.append(patches)
    return data, demograp_data

def train(model, X_train: torch.tensor, y_train: torch.tensor, cv,
          batch_size:int , epochs: int, device,):
    # Entrenamiento por mini-lotes
    accuracy_test=0

    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_train.cpu(), y_train.cpu())):
        model =model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=2e-4)
        steps_per_epoch = len(train_idx) // batch_size
        total_steps = steps_per_epoch * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=max(1, total_steps))
        Acc_scores = []
        Loss_scores=[]
        
        for epoch in range(epochs):
            model.train()
            idx = train_idx[torch.randperm(len(train_idx))]
            for i in range(0, len(idx)- batch_size):
                batch_idx = idx[i:i + batch_size]

                X_batch = X_train[batch_idx].float().to(device)
                #y_batch = y_train[batch_idx].float().to(device)
                y_batch = y_train[batch_idx].long().to(device)
                
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            model.eval()
            with torch.no_grad():
                Loss_scores.append(loss.item())
                pred = outputs.argmax(dim=1)
                acc_train = (pred == y_batch).float().mean().item()
                #pred = torch.argmax(model(X_train[train_idx].float().to(device)), dim=1)
                #acc_train = (pred == y_train[train_idx].long().to(device)).float().mean().item()
                Acc_scores.append(acc_train)

                if epoch % (epochs//2)== 0:
                        print(f"[Fold {fold+1}] Época {epoch}: pérdida={loss.item():.4f}, acc_train={acc_train:.3f}")
                
        

        model.eval()
        with torch.no_grad():
            logits = model(X_train[test_idx].float().to(device))
            preds = torch.argmax(torch.softmax(logits.float().to(device), dim=1), dim=1)
            acc = (preds == y_train[test_idx].to(device)).float().mean().item()
            accuracy_test+=acc
            print(f"[Fold {fold+1}] Precisión validación: {acc:.3f}")

    
    print("Average kfold accuracy: ", accuracy_test / len(list(cv.split(X_train.cpu(), y_train.cpu()))))
    return model, Loss_scores, Acc_scores