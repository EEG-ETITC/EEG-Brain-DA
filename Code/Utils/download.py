import requests
import numpy as np
import pandas as pd
from scipy.io import loadmat
import mne
import os
import pickle
from scipy.stats import pearsonr
from torch import from_numpy
import warnings
warnings.filterwarnings('ignore')


def load_signal(mat_file, hea_file):
    mat=loadmat(mat_file)
    signal=mat["val"] #  mat["n_channels"], mat["n_samples"]

    # Leer el archivo .hea
    with open(hea_file, 'r') as f:
        lines=f.readlines()
    header_main= lines[0].strip().split()
    num_channels=int(header_main[1])

    # Frecuencia de muestreo  (fs) esta en la primera linea, en el tercer campo
    fs=float(lines[0].split()[2])

    channel_lines=[line for line in lines[1:] if len(line.split())>=9]

    # nombre del canal, 9no campo en cada linea valida
    channels=[line.split()[8] for line in channel_lines]

    assert signal.shape[0]==len(channels), f'Canales en {mat_file} no coinciden con .hea'

    return {
       'signal':signal,
       'channels': channels,
       'fs': fs 
    }


def create_raw_array(signal, channels, sfreq, group):
    eeg_names = ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'C3', 'C4',
                 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz', 'Fpz', 'Oz', 'F9']
    ecg_names = ['ECG', 'ECG1', 'ECG2', 'ECGL', 'ECGR']

    ch_types = []
    for ch in channels:
        if ch in eeg_names:
            ch_types.append('eeg')
        elif ch in ecg_names:
            ch_types.append('ecg')
        else:
            ch_types.append('misc')

    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=ch_types)

    # Solo escalar si el grupo es EEG (o si el canal es eeg)
    if all(t == 'eeg' for t in ch_types):
        raw = mne.io.RawArray(signal * 1e-6, info, verbose=False)
    else:
        raw = mne.io.RawArray(signal, info, verbose=False)

    raw.info['description'] = f'{group} data'
    return raw



def download_signal_EEG(data_patients, hour_list:list, download_paht:str, format:str="EEG",
                        login_url:str="https://physionet.org/login/", 
                        payload:dict  = {"username": "juanlealc", "password": "t0sc0f0nd0blanc0"},
                        k_proving:int=None, remove_data=False, lowcut=0.5, highcut=45, save_data_path:str=None,
                        interval_times:list[tuple[int,int]] = None, fs_resampled:int=None, interpolate:bool=True, get_errors:bool=False)->np.array:
    df_filtered=pd.DataFrame(columns=data_patients.columns)
    for  h in hour_list:
        df_filtered=pd.concat([df_filtered , data_patients[(data_patients["hour_file"]==h) & (data_patients["format_data"]==format)]])    
    patients=np.unique(df_filtered["patient"])
    
    if type(k_proving)==int:
        patients=patients[:k_proving]
        
    with requests.Session() as s:
        s.post(login_url, data=payload)
        raw_list=[]
        ref_list=[]
        rejected=[]
        if get_errors:
            errors_interpolate={}
        for patient in patients:
            df_aux=df_filtered[df_filtered["patient"]==patient]
            
            for hour in pd.unique(df_aux["hour_file"]):
                
                if hour in hour_list:
                    ref_list.append((int(patient), hour))
                    path_list=df_aux["path"][df_aux["hour_file"]==hour].tolist()
                    
                    if len(path_list)==1:
                        path_mat=path_list[0]+".mat"
                        path_hea=path_list[0]+".hea"
                        patient=path_list[0].split("/")[1]
                        print("________________________________________")
                        print(f"downloading from path: {path_mat}...")
                        print(f"downloading from path: {path_hea}...")
                        print(patient)
                        try:
                            url_hea = f"https://physionet.org/static/published-projects/i-care/2.1/{path_hea}"
                            resp_hea = s.get(url_hea)
                            resp_hea.raise_for_status()
                
                            with open(f"{download_paht}/{patient}_{hour}.hea", "wb") as f:
                                f.write(resp_hea.content)
     
                            url_mat = f"https://physionet.org/static/published-projects/i-care/2.1/{path_mat}"
                            resp_mat = s.get(url_mat)
                            resp_mat.raise_for_status()
                        
                            with open(f"{download_paht}/{patient}_{hour}.mat", "wb") as f:
                                f.write(resp_mat.content)
                        
                            mat_path=f"{download_paht}/{patient}_{hour}.mat"
                            hea_path=f"{download_paht}/{patient}_{hour}.hea"
                            signal_data = load_signal(mat_path, hea_path)
                            final_raw = create_raw_array(signal_data['signal'], 
                                                    signal_data['channels'], signal_data['fs'], format)
                            final_raw=final_raw.copy().filter(lowcut, highcut, fir_design='firwin')
                            
                            if interpolate:
                                errors = {}
                                for ch in final_raw.ch_names:
                                    errors[ch] = interpolation_error(raw=final_raw, ch_name=ch)
                                
                                if get_errors:
                                    errors_interpolate[patient+str(hour)]=errors
                                
                                threshold = np.mean(list(errors.values())) + 2.5 * np.std(list(errors.values()))
                                bad_channels = [ch for ch, err in errors.items() if err > threshold]
                                # Marcar canales malos
                                if len(bad_channels)<=3:
                                    final_raw.info["bads"] = bad_channels
                                    # # Interpolar
                                    final_raw.interpolate_bads(reset_bads=True)
                                else:
                                    print(f"Becarefull with the patient {patient}, at the hour {hour}")
                            
                            
                            if interval_times is not None and fs_resampled is not None:
                                raw_aux=data_time_to_load(raw=final_raw, intervals=interval_times, fs_resampled=fs_resampled)
                            
                            if raw_aux is not None:
                                raw_list.append(raw_aux)
                            else:
                                ref_list.remove((int(patient), hour))
                                rejected.append((int(patient), hour))
                                del errors_interpolate[patient+str(hour)]    


                        except NameError as e:
                            print("Error", e)
                            print("check the number of the file and the time")
                    elif len(path_list)>1:
                        raws=list()
                        for k, path in enumerate(path_list):
                            path_mat=path+".mat"
                            path_hea=path+".hea"
                            patient=path.split("/")[1]
                            print("________________________________________")
                            print(f"downloading from path: {path_mat} part {k+1}...")
                            print(f"downloading from path: {path_hea} part {k+1}...")
                            print(patient)
                            try:
                                url_hea = f"https://physionet.org/static/published-projects/i-care/2.1/{path_hea}"
                                resp_hea = s.get(url_hea)
                                resp_hea.raise_for_status()
                    
                                with open(f"{download_paht}/{patient}_{hour}_{k+1}.hea", "wb") as f:
                                    f.write(resp_hea.content)
        
                                url_mat = f"https://physionet.org/static/published-projects/i-care/2.1/{path_mat}"
                                resp_mat = s.get(url_mat)
                                resp_mat.raise_for_status()
                            
                                with open(f"{download_paht}/{patient}_{hour}_{k+1}.mat", "wb") as f:
                                    f.write(resp_mat.content)
                            
                                mat_path=f"{download_paht}/{patient}_{hour}_{k+1}.mat"
                                hea_path=f"{download_paht}/{patient}_{hour}_{k+1}.hea"
                                signal_data = load_signal(mat_path, hea_path)
                                raw = create_raw_array(signal_data['signal'], 
                                                        signal_data['channels'], signal_data['fs'], format)
                                raws.append(raw)
                                
                            except NameError as e:
                                print("Error", e)
                                print("check the number of the file and the time")
                                
                        if len(np.unique([raw.info['sfreq'] for raw in raws]))>1:
                            fs_res=np.sort(np.unique([raw.info['sfreq'] for raw in raws]))[0]
                            raws=[raw.resample(fs_res) if raw.info['sfreq']!=fs_res else raw for raw in raws ]
                            
                        try:
                            final_raw=mne.concatenate_raws(raws=raws)
                        except:
                            try:
                                raws= mne.match_channel_orders(raws, copy=True)    
                                final_raw=mne.concatenate_raws(raws=raws)
                            except:
                                final_raw=raws[0]
                        
                        final_raw=final_raw.copy().filter(lowcut, highcut, fir_design='firwin')
                        
                        if interpolate:
                                errors = {}
                                for ch in final_raw.ch_names:
                                    errors[ch] = interpolation_error(raw=final_raw, ch_name=ch)
                                
                                if get_errors:
                                    errors_interpolate[patient+str(hour)]=errors
                                    
                                threshold = np.mean(list(errors.values())) + 2.5 * np.std(list(errors.values()))
                                bad_channels = [ch for ch, err in errors.items() if err > threshold]
                                # Marcar canales malos
                                if len(bad_channels)<=3:
                                    final_raw.info["bads"] = bad_channels
                                    # # Interpolar
                                    final_raw.interpolate_bads(reset_bads=True)
                                else:
                                    print(f"Becarefull with the patient {patient}, at the hour {hour}")
                        
                        if interval_times is not None and fs_resampled is not None:
                                raw_aux=data_time_to_load(raw=final_raw, intervals=interval_times, fs_resampled=fs_resampled)
                        
                        if raw_aux is not None:
                            raw_list.append(raw_aux)
                        else:
                            ref_list.remove((int(patient), hour))
                            rejected.append((int(patient), hour))
                            del errors_interpolate[patient+str(hour)]
                else:
                    print(f" the hour {hour} is not in the list")
            if remove_data:
                data_list=os.listdir(download_paht)
                print("Lista de datos a eliminar")
                print(data_list)
                for data in data_list:
                    try:
                        os.remove(download_paht+data)
                        print(f"Archivo '{data}' eliminado correctamente.")
                    except FileNotFoundError:
                        print(f"El archivo '{data}' no existe.")
                    except OSError as e:
                        print(f"Error: {e.strerror} - {e.filename}")
    
    if save_data_path is not None and type(save_data_path)==str:
        # Guardar
        with open(save_data_path+'datos.pkl', 'wb') as f:
            pickle.dump(raw_list, f)
    if get_errors:
        try:
            return raw_list, ref_list, errors_interpolate, rejected
        except:
            return None, None, None, None
    else:
        try:
            return raw_list, ref_list, rejected
        except:
            return None, None, None
        


def data_time_to_load(raw: list, intervals: list[tuple[int,int]],  fs_resampled: int=None)->list:
    """Intervals: it has to take a list of tuples, each one have to composed of two entire numbers where the first is the min value of the
    interval and the second one is the max value, if it is necessary to take more tha one interval, each one will have to have the 
    same number min and max value, take into account the time represented in the signal. The interval have to be more than one second 
    and the unit measure of the value has to be seconds. Example [(0,10), (5,15), (20,30), (1200,1300)]"""

    if fs_resampled is None:
        signal_points=raw.copy().get_data()
        fs=raw.info["sfreq"]
    
    else:
        info=raw.copy().resample(fs_resampled).info
        signal_points=raw.copy().resample(fs_resampled).get_data()
        fs=fs_resampled

    if type(intervals) != list:
        return "Ckeck the tuple be in a list, even if it's just one"
    
    signal_shape=signal_points.shape
    seconds_cant=int(np.floor(signal_shape[1]/fs))
    print(f"The signal has a quantity of {seconds_cant} seconds")
    singal_intervaled=[]
    for interval in intervals:
        min_interval=int(interval[0]*fs)
        max_interval=int(interval[1]*fs)
        if min_interval > signal_shape[1] or max_interval > signal_shape[1]:
            print("-"*5)
            print("-"*8)
            print("-"*10)
            print(f"Cannot take the interval {interval}, check the time duration of the signal")
            print("-"*10)
            print("-"*8)
            print("-"*5)
            
        else:
            print(f"Interval {interval} seconds added to the data")
            singal_intervaled.append(signal_points[:,min_interval:max_interval])
    try:
        return mne.io.RawArray(data=np.concat(singal_intervaled, axis=1), info=info)
    except:
        return None


def paching(data, window_size: int, overlapping_size: int):
    patches = []
    pos=[]
    start = 0
    end=window_size
    patches.append(data[:,start:end])
    pos.append((start,end))
    T=data.shape[1]
    
    while end < T:
        
        start = start + window_size - overlapping_size
        end = start + window_size
        
        if end > T:
            end = T
            start = T - window_size
            pos.append((start,end))
            patches.append(data[:, start:end])
            break
        pos.append((start,end))
        patches.append(data[:, start:end])
    
    patches=np.stack(patches)
    return from_numpy(patches), pos



def download_txt(path, save:bool, path_save: str):
    df_times=pd.read_csv(path,sep=" ",header=None)
    df_times.drop(index=[0, 1, 162834],inplace=True)
    df_times[2]= df_times[1].str.split(pat="/").apply(lambda x: x[0])
    df_times[3]= df_times[1].str.split(pat="/").apply(lambda x: x[1])
    df_times[4]= df_times[1].str.split(pat="/").apply(lambda x: x[2])
    df_times[5]= df_times[4].str.split(pat="_").apply(lambda x: x[1] if len(x)>=2 else np.nan)
    df_times[6]= df_times[4].str.split(pat="_").apply(lambda x: x[2] if len(x)>=3 else np.nan)
    df_times[7]= df_times[4].str.split(pat="_").apply(lambda x: x[3].split(sep=".")[0] if len(x)>3 else np.nan)
    df_times[8]= df_times[4].str.split(pat=".").apply(lambda x: x[-1])
    df_times.columns=["id_path", "path", "data_partition", "patient", "file", "number_file", "hour_file", 
                 "format_data", "file_extension"]
    df_times["meas_date"]=pd.Series()
    df_times["minutes"]=pd.Series()
    df_times["#_channels"]=pd.Series()
    df_times["channels"]=pd.Series()
    df_times["fs"]=pd.Series()
    df_times["time_points"]=pd.Series()
    
    if save:
        df_times.to_csv(path_or_buf=path_save, sep=";", index=False)
    return df_times


def interpolation_corr(raw, ch_name):
    raw_copy = raw.copy()
    x_orig = raw_copy.get_data(picks=[ch_name])[0]

    raw_copy.info["bads"] = [ch_name]
    raw_copy.interpolate_bads(reset_bads=True)
    x_interp = raw_copy.get_data(picks=[ch_name])[0]

    r, _ = pearsonr(x_orig, x_interp)
    return r

    
def interpolation_error(raw, ch_name):
    """
    Calcula el error entre el canal original y su versión interpolada
    """
    raw.set_montage("standard_1020")
    raw.pick_types(eeg=True)
    raw_copy = raw.copy()

    # Señal original
    x_orig = raw_copy.get_data(picks=[ch_name])[0]

    # Marcar canal como malo
    raw_copy.info["bads"] = [ch_name]

    # Interpolar
    raw_copy.interpolate_bads(reset_bads=True)

    # Señal interpolada
    x_interp = raw_copy.get_data(picks=[ch_name])[0]

    # Error RMS normalizado
    error = np.sqrt(np.mean((x_orig - x_interp) ** 2)) / (np.std(x_orig)+1e-6)

    return error