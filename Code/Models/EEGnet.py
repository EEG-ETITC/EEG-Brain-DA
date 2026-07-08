import sys
import os
 
# Obtener la ruta absoluta del directorio padre
# sys.path.append('../')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from seaborn import heatmap
from Utils.train import  patient_stratify_split, cv_patients


from skorch.callbacks import LRScheduler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from braindecode import EEGClassifier
from braindecode.models import EEGNet
from braindecode.util import set_random_seeds
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc


# Set random seed to ensure reproducible initialization below
seed = 2024
cuda = torch.cuda.is_available()
set_random_seeds(seed=seed, cuda=cuda)

channels=['Fp1', 'Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']
channels_dif=['Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'F7-T3', 'T3-T5', 'T5-O1', 'F8-T4', 'T4-T6', 'T6-O2', 'Fz-Cz', 'Cz-Pz']
path_demographic="Code/d/data_patients.csv"

### Data base demographic
df_demog=pd.read_csv(path_demographic, sep="," )
df_demog["numeric_outcome"]=df_demog["Outcome"].map({"Good": 1, "Poor": 0})
df_demog.reset_index(drop=True, inplace=True)

### Ruta de donde provienen los archivos .fif, tener en cuenta que esta ruta llama las carpetas por hora.
path_files="Code/raw_tif/"
hours=[32, 36, 40, 44 ]
hour_list = os.listdir(path_files)
hour_list = [x for x in hour_list if int(x.split("h")[1]) in hours]
Raws=[]
Patients=[]
# Carga Pre-Launch de MNE asegurándose que la RAM los acople sin retrasos lazy (preload=True)
for h in hour_list:
    print(h)
    file_list = os.listdir(path_files+h+"/")
    Raws += [mne.io.read_raw_fif(path_files+h+"/"+file, preload=True, verbose=0) for file in file_list]
    Patients += [file.split(".")[0][-4:] for file in file_list]

X=[]
Scaler=StandardScaler()
demograp_data=pd.DataFrame(columns=df_demog.columns)
for k, raw in enumerate(Raws):
    patient=Patients[k]
    demograp_data=pd.concat([demograp_data,df_demog[df_demog["Id_Patient"]==int(patient)]])
    X.append(Scaler.fit_transform(raw.get_data(picks=channels)))
del Raws
X=np.c_[X]
Y=np.int_(demograp_data["numeric_outcome"].values)
patients_train, patients_val= patient_stratify_split(demograp_data, train_size=0.75)
# X_train, Y_train=X[patients_train], Y[patients_train]
X_val, Y_val=X[patients_val], Y[patients_val]
n_times=X[patients_train].shape[-1]
n_chans=X[patients_train].shape[1]
outputs=np.unique(Y[patients_train])
S_labels=["Poor", "Good"]
# del X, Y


model = EEGNet(n_times=n_times, n_chans=n_chans, n_outputs=len(outputs), activation=torch.nn.GELU, drop_prob=0.1)
model.cuda()
lr = [6e-6, 1e-6]
weight_decay = [1e-7]
batch_size = 64
n_epochs = 25

parameters={"optimizer__lr":lr, "optimizer__weight_decay":weight_decay, "module__drop_prob":np.linspace(0, 0.2, 2),
            "module__activation": [torch.nn.RReLU], "module__final_conv_length": ["auto", 512],
            "module__F1": [4, 8, 12], "module__D": [2, 4], "module__kernel_length": [32, 64],
            "module__pool1_kernel_size": [4, 8], "module__pool2_kernel_size": [ 4, 8], "module__batch_norm_affine": [True],
            "module__n_times": [n_times], "module__n_chans":[n_chans], "module__n_outputs":[len(outputs)], "module__sfreq":[128]}


clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    train_split=None,
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "roc_auc",
        "f1",
        "recall",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
    ],
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    classes=outputs,
    max_epochs=n_epochs,
    verbose=2,
)

grid_clf=GridSearchCV(clf, parameters, refit=True, cv=cv_patients(patients_train, n_split=3), scoring="roc_auc", verbose=1)

grid_clf.fit(X, Y)

# evaluated the model after training
y_pred= grid_clf.predict(X_val)
report = classification_report(Y_val, y_pred,target_names=["Poor Outcome", "Good Outcome"])
confusion = confusion_matrix(Y_val, y_pred,normalize="true")
print(report)
plt.figure()
heatmap(confusion, annot=True, fmt=".2g",
        xticklabels=S_labels,
        yticklabels=S_labels)


plt.title(f"EEGnet model")
plt.savefig(f"EEGnet_model_confusion_matrix.png", dpi=300)
plt.show()


y_score = grid_clf.predict_proba(X_val)
fpr, tpr, thresholds = roc_curve(Y_val, y_score[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, lw=2,
            label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'Curva ROC {S_labels[1]}')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(f"EEGnet_model_ROC_curve.png", dpi=300)
plt.show()