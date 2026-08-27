import requests
import numpy as np
import pandas as pd
from scipy.io import loadmat
import mne
import os
import pickle
from scipy.stats import pearsonr
from torch import from_numpy
from scipy.signal import butter, filtfilt
from Utils.bad_channels import NoisyChannels
import warnings
import matplotlib.pyplot as plt
from Utils.utils import _eeglab_interpolate_bads
warnings.filterwarnings('ignore')


# Montaje Bipolar Clínico (Banana)
# Este montaje reduce el ruido de campo lejano y resalta asimetrías locales.
ANODE = [
    'Fp1', 'F3', 'C3', 'P3',  # Cadena Parasagital Izquierda
    'Fp2', 'F4', 'C4', 'P4',  # Cadena Parasagital Derecha
    'F7', 'T3', 'T5',         # Cadena Temporal Izquierda
    'F8', 'T4', 'T6',         # Cadena Temporal Derecha
    'Fz', 'Cz'                # Cadena Central (opcional)
]
CATHODE = [
    'F3', 'C3', 'P3', 'O1',   # Cadena Parasagital Izquierda
    'F4', 'C4', 'P4', 'O2',   # Cadena Parasagital Derecha
    'T3', 'T5', 'O1',         # Cadena Temporal Izquierda
    'T4', 'T6', 'O2',         # Cadena Temporal Derecha
    'Cz', 'Pz',               # Cadena Central (opcional)
]
CH_NAMES = [f'{a}-{c}' for a, c in zip(ANODE, CATHODE)]


import ctypes

def empty_recycle_bin():
    SHERB_NOCONFIRMATION = 0x00000001
    SHERB_NOPROGRESSUI = 0x00000002
    SHERB_NOSOUND = 0x00000004

    result = ctypes.windll.shell32.SHEmptyRecycleBinW(
        None,
        None,
        SHERB_NOCONFIRMATION | SHERB_NOPROGRESSUI | SHERB_NOSOUND
    )

    if result == 0:
        print("Recycle Bin emptied successfully.")
    else:
        print("Failed to empty Recycle Bin. Error code:", result)


def load_signal(mat_file, hea_file):
    """
    Carga una señal EEG desde el formato WFDB (.mat y .hea) de PhysioNet.

    Parameters
    ----------
    mat_file : str
        Ruta del archivo .mat descargado que contiene la matriz de datos de la señal.
    hea_file : str
        Ruta del archivo .hea descargado que contiene los metadatos y configuración de canales.

    Returns
    -------
    dict
        Diccionario con la siguiente información:
        - 'signal' (np.ndarray): Matriz continua de datos numéricos (n_channels, n_samples).
        - 'channels' (list[str]): Lista ordenada con los nombres de los canales.
        - 'fs' (float): Frecuencia de muestreo de la señal.

    Raises
    ------
    AssertionError
        Si la cantidad de canales listados en el .hea no coincide con la dimensión de la matriz .mat.
    """
    # Leer el archivo .mat con scipy
    mat = loadmat(mat_file)
    signal = mat["val"] # Dimensiones esperadas: [n_channels, n_samples]

    # Leer el archivo de encabezado (.hea) para extraer metadatos
    with open(hea_file, 'r') as f:
        lines = f.readlines()
    
    header_main = lines[0].strip().split()
    num_channels = int(header_main[1])

    # La frecuencia de muestreo (fs) se encuentra en la primera línea, en el tercer campo
    fs = float(lines[0].split()[2])

    channel_lines = [line for line in lines[1:] if len(line.split()) >= 9]

    # Nombre del canal, 9no campo en cada línea válida del formato I-CARE
    channels = [line.split()[8] for line in channel_lines]

    assert signal.shape[0] == len(channels), f'Canales en {mat_file} no coinciden con .hea'

    return {
       'signal': signal,
       'channels': channels,
       'fs': fs 
    }


def create_raw_array(signal, channels, sfreq, group):
    """
    Construye un objeto RawArray de MNE, escalado y clasificado por el tipo de canal.

    Parameters
    ----------
    signal : np.ndarray
        Matriz bidimensional con los datos recogidos [n_channels, n_samples].
    channels : list of str
        Nombres identificadores extraídos de los electrodos, asignados a posiciones estándar.
    sfreq : float
        Frecuencia de muestreo para la señal continua.
    group : str
        Etiqueta de descripción acerca del registro o tipo de formato.

    Returns
    -------
    mne.io.RawArray
        Instancia en bruto lista para aplicar tuberías de procesamiento temporal y espacial con MNE.
    """
    # Listas estandarizadas para mapeo de tipo de canal
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

    # Solo escalar la unidad fisiológica (a Voltios) si todos los canales detectados son puramente EEG
    if all(t == 'eeg' for t in ch_types):
        raw = mne.io.RawArray(signal * 1e-6, info, verbose=False)
    else:
        raw = mne.io.RawArray(signal, info, verbose=False)

    raw.info['description'] = f'{group} data'
    return raw



def download_signal_EEG(data_patients, hour_list: list, download_paht: str, format: str = "EEG",
                        login_url: str = "https://physionet.org/login/", 
                        payload: dict = {"username": "juanlealc", "password": "t0sc0f0nd0blanc0"},
                        k_proving: int = None, remove_data=False, lowcut=0.5, highcut=45, save_data_path: str = None,
                        interval_times: list[tuple[int,int]] = None, fs_resampled: int = None, interpolate: bool = True,
                        top_map_save_path: str = None, raw_save_path: str = None) -> np.array:
    """
    Descarga registros EEG/ECG de la base de datos I-CARE en PhysioNet, aplica filtrado FIR, referencias
    estándar, interpolado espacial y segmentado temporal de la señal lista para Machine Learning.

    Parameters
    ----------
    data_patients : pd.DataFrame
        DataFrame con las rutas (path) y metadatos de los pacientes alojados en PhysioNet.
    hour_list : list
        Lista de identificadores de horas específicas requeridas para descargar.
    download_paht : str
        Ruta local para almacenar (volátil o permanentemente) los archivos .mat y .hue nativos descargados.
    format : str
        Formato destino asignado al grupo extraído (default: "EEG").
    login_url : str
        Endpoint de autenticación de PhysioNet.
    payload : dict
        Credenciales dePhysioNet requeridas para datos de acceso restringido o credencializados.
    k_proving : int, opcional
        Límite de pacientes a operar (útil para pruebas de prototipado rápido).
    remove_data : bool, opcional
        Si es True, elimina todos los archivos en `download_paht` después del procesamiento general.
    lowcut : float, opcional
        Límite inferior para el filtro paso banda FIR en hertzios (default: 0.5Hz, remueve deriva basal).
    highcut : float, opcional
        Límite superior para el filtro paso banda FIR en hertzios (default: 45Hz, remueve ruido eléctrico).
    save_data_path : str, opcional
        Ruta local final donde se guardará (vía Pickle) la lista resultante de matrices MNE Raw.
    interval_times : list of tuple, opcional
        Segmentos específicos a recortar representados en tuplas (segundo_inicio, segundo_fin).
    fs_resampled : int, opcional
        Frecuencia destino a la que los recortes se submuestrearán dinámicamente.
    interpolate : bool, opcional
        Habilita la estimación topográfica 'spline' esférica para subsanar los canales malos detectados.
    top_map_save_path : str, opcional
        Directorio donde guardar imágenes PNG del Densidad Espectral de Potencia Mapeada (PSD Topomap).
    raw_save_path_ str 

    Returns
    -------
    tuple
        Una tupla que conforma de forma general:
        - raw_list (list): Instancias `mne.io.RawArray` filtradas, montadas y epocadas.
        - ref_list (list): Pares de indexación con metadatos asociados a `(paciente, hora)`.
        - (optional) errors_interpolate: Registro auxiliar de errores algorítmicos.
        - rejected (list): Tuplas de pacientes procesados con error u omitidos del conjunto final.
    """
    # 1. Filtrar lista global de pacientes basado en los argumentos recibidos
    df_filtered = pd.DataFrame(columns=data_patients.columns)
    for h in hour_list:
        df_filtered = pd.concat([df_filtered, data_patients[(data_patients["hour_file"] == h) & (data_patients["format_data"] == format)]])    
    patients = np.unique(df_filtered["patient"])
    
    # Creación de directorio local para almacenamiento de los raw tipo .fit descargados y procesados, si se especificó una ruta válida
    if raw_save_path is not None and type(raw_save_path) == str:
        os.makedirs(raw_save_path, exist_ok=True)
    
    # Acortar la ejecución (pruebas o debug)
    if type(k_proving) == int:
        patients = patients[:k_proving]
        
    with requests.Session() as s:
        # 2. Iniciar sesión autorizada en PhysioNet
        s.post(login_url, data=payload)
        # raw_list = []
        ref_list = []
        rejected = []
            
        # 3. Iterar cada paciente validado
        for patient in patients:
            
            df_aux = df_filtered[df_filtered["patient"] == patient]
            
            for hour in pd.unique(df_aux["hour_file"]):
                
                if hour in hour_list:
                    ref_list.append((int(patient), hour))
                    path_list = df_aux["path"][df_aux["hour_file"] == hour].tolist()
                    
                    # CASO A: Existe una única ruta/archivo mapeado para la hora
                    if len(path_list) == 1:
                        path_mat = path_list[0] + ".mat"
                        path_hea = path_list[0] + ".hea"
                        patient = path_list[0].split("/")[1]
                        print("________________________________________________________________________________________________________________________")
                        print("________________________________________________________________________________________")
                        print("___________________________________________________________________________")
                        print(f"Descargando desde la ruta: {path_mat}...")
                        print(f"Descargando desde la ruta: {path_hea}...")
                        print(patient)
                        try:
                            # Descargar e Instanciar Headers
                            url_hea = f"https://physionet.org/static/published-projects/i-care/2.1/{path_hea}"
                            resp_hea = s.get(url_hea)
                            resp_hea.raise_for_status()
                
                            with open(f"{download_paht}/{patient}_{hour}.hea", "wb") as f:
                                f.write(resp_hea.content)
     
                            # Descargar e Instanciar Señales Matriciales
                            url_mat = f"https://physionet.org/static/published-projects/i-care/2.1/{path_mat}"
                            resp_mat = s.get(url_mat)
                            resp_mat.raise_for_status()
                        
                            with open(f"{download_paht}/{patient}_{hour}.mat", "wb") as f:
                                f.write(resp_mat.content)
                        
                            mat_path = f"{download_paht}/{patient}_{hour}.mat"
                            hea_path = f"{download_paht}/{patient}_{hour}.hea"
                            
                            # Fusionar MNE y RawArray
                            signal_data = load_signal(mat_path, hea_path)
                            raw = create_raw_array(signal_data['signal'], 
                                                    signal_data['channels'], signal_data['fs'], format)
                            
                            # 4. PREPROCESAMIENTO CLÍNICO
                            # 4.1. Filtrado FIR pasa banda (default 0.5-45 Hz)
                            raw = raw.copy().filter(lowcut, highcut, fir_design='firwin')
                            
                            # 4.2. Asignación espacial estándar de electrodos
                            raw.set_montage("standard_1020") 
                            
                            # 4.3. Instanciar clase de detección automática de Canales Ruidosos / Artefactuales
                            denoisier = NoisyChannels(raw=raw, random_state=23)
                            denoisier.find_all_bads()
                            
                            if denoisier.get_bads() is not None:
                                raw.info["bads"] = denoisier.get_bads()
                            
                                # 4.4. Setear una nueva referencia global usando el promedio común
                                raw.set_eeg_reference("average")
                                
                                # 4.5. Extraer Mapas de frecuencias Power Spectral Density (Si se pidió)
                                if top_map_save_path is not None and type(top_map_save_path) == str:
                                    try:
                                        fig = raw.copy().resample(128).plot_psd_topomap(show=False)
                                        fig.savefig(top_map_save_path + f"topomap_psd_{patient}_{hour}.png", dpi=300, bbox_inches="tight")
                                        plt.close(fig)
                                    except Exception as e:
                                        print(f"Error saving topomap for the patient {patient} in hour {hour}: {e}")
                                        
                                # 4.6. Rellenar o reparar canales malos usando método topológico eeglab de spines.
                                if interpolate:
                                    _eeglab_interpolate_bads(raw=raw)

                                # 4.7. Recalcular derivaciones hacia el Montaje Bipolar ("Doble Banana") local para la Red Neuronal
                                final_raw = mne.set_bipolar_reference(raw, anode=ANODE, cathode=CATHODE, ch_name=CH_NAMES, drop_refs=False).copy()
                                
                                
                                # 4.8. Cortar / Aislar un segmento de tiempo de interés y remuestrearlo a `fs_resampled`
                                if interval_times is not None and fs_resampled is not None:
                                    final_raw = data_time_to_load(raw=final_raw, intervals=interval_times, fs_resampled=fs_resampled)
                                
                                if final_raw is not None:
                                    # raw_list.append(final_raw)
                                    if raw_save_path is not None and type(raw_save_path) == str :
                                        final_raw.save(raw_save_path+"patient_"+str(patient)+".fif",overwrite=True) 
                                        print("________________________________________________________________________________________________________________________")
                                        print("________________________________________________________________________________________________________________________")
                                        print("________________________________________________________________________________________________________________________")
                                        print(f"File 'patient_{patient}.fif' in hour {hour} saved correctly in {raw_save_path}.")    
                                        print("________________________________________________________________________________________________________________________")
                                        print("________________________________________________________________________________________________________________________")
                                        print("________________________________________________________________________________________________________________________")
                                    del raw, final_raw
                                else:
                                    ref_list.remove((int(patient), hour))
                                    rejected.append((int(patient), hour))
                                    print("________________________________________________________________________________________________________________________")
                                    print("________________________________________________________________________________________________________________________")
                                    print("________________________________________________________________________________________________________________________")
                                    print(f"The patient {patient} in hour {hour} was rejected during preprocessing.")
                                    print("________________________________________________________________________________________________________________________")
                                    print("________________________________________________________________________________________________________________________")
                                    print("________________________________________________________________________________________________________________________")
                                
                                
                            # Retiro de los datos procesados que no pasaron el proceso de filtrado, referencia o epocaje, y registro de los casos rechazados
                            else:
                                ref_list.remove((int(patient), hour))
                                rejected.append((int(patient), hour))
                                print("________________________________________________________________________________________________________________________")
                                print("________________________________________________________________________________________________________________________")
                                print("________________________________________________________________________________________________________________________")
                                print(f"The patient {patient} in hour {hour} was rejected during preprocessing.")
                                print("________________________________________________________________________________________________________________________")
                                print("________________________________________________________________________________________________________________________")
                                print("________________________________________________________________________________________________________________________")
                                
                        
                        except NameError as e:
                            print("Error NameError encontrado:", e)
                            print("Verify the file number split from Physionet")
                        
                            
                            
                    # CASO B: Existen múltiples rutas continuas asociadas al mismo registro
                    elif len(path_list) > 1:
                        raws = list()
                        for k, path in enumerate(path_list):
                            path_mat = path + ".mat"
                            path_hea = path + ".hea"
                            patient = path.split("/")[1]
                            print("________________________________________________________________________________________________________________________")
                            print("________________________________________________________________________________________")
                            print("___________________________________________________________________________")
                            print(f"Descargando desde la ruta: {path_mat} parte {k+1}...")
                            print(f"Descargando desde la ruta: {path_hea} parte {k+1}...")
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
                            
                                mat_path = f"{download_paht}/{patient}_{hour}_{k+1}.mat"
                                hea_path = f"{download_paht}/{patient}_{hour}_{k+1}.hea"
                                
                                signal_data = load_signal(mat_path, hea_path)
                                raw_aux = create_raw_array(signal_data['signal'], 
                                                        signal_data['channels'], signal_data['fs'], format)
                                raws.append(raw_aux)
                                del raw_aux
                                
                            except NameError as e:
                                print("Error NameError encontrado:", e)
                                print("Verifique el número del archivo partido de Physionet")
                                
                        # Resolver variaciones de Frecuencia de muestreo (fs) si los fragmentos vienen dispares
                        if len(np.unique([raw.info['sfreq'] for raw in raws])) > 1:
                            fs_res = np.sort(np.unique([raw.info['sfreq'] for raw in raws]))[0]
                            raws = [raw.resample(fs_res) if raw.info['sfreq'] != fs_res else raw for raw in raws]
                            
                        # Concatenar todos los fragmentos
                        try:
                            raw = mne.concatenate_raws(raws=raws)
                        except:
                            # Forzar uniformidad topológica como plan de contingencia si difieren orden de canales
                            try:
                                raws = mne.match_channel_orders(raws, copy=True)    
                                raw = mne.concatenate_raws(raws=raws)
                            except:
                                raw = raws[0] # Fallback Extremo: Desechar cortes y conservar solo el de cabeza
                        
                        ### Filtrado de la señal FIR
                        raw = raw.copy().filter(lowcut, highcut, fir_design='firwin')
                        
                        ## Referenciar montaje geométrico
                        raw.set_montage("standard_1020")
                        
                        ## Identificar canales malos algorítmicamente
                        denoisier = NoisyChannels(raw=raw, random_state=23)
                        denoisier.find_all_bads()
                        
                        if denoisier.get_bads() is not None:
                            raw.info["bads"] = denoisier.get_bads()
                        
                            # 4.4. Setear una nueva referencia global usando el promedio común
                            raw.set_eeg_reference("average")
                            
                            # 4.5. Extraer Mapas de frecuencias Power Spectral Density (Si se pidió)
                            if top_map_save_path is not None and type(top_map_save_path) == str:
                                try:
                                    fig = raw.copy().resample(128).plot_psd_topomap(show=False)
                                    fig.savefig(top_map_save_path + f"topomap_psd_{patient}_{hour}.png", dpi=300, bbox_inches="tight")
                                    plt.close(fig)
                                except Exception as e:
                                    print(f"Error saving topomap for the patient {patient} in hour {hour}: {e}")
                                    
                            # 4.6. Rellenar o reparar canales malos usando método topológico eeglab de spines.
                            if interpolate:
                                _eeglab_interpolate_bads(raw=raw)

                            # 4.7. Recalcular derivaciones hacia el Montaje Bipolar ("Doble Banana") local para la Red Neuronal
                            final_raw = mne.set_bipolar_reference(raw, anode=ANODE, cathode=CATHODE, ch_name=CH_NAMES, drop_refs=False).copy()
                            
                            
                            # 4.8. Cortar / Aislar un segmento de tiempo de interés y remuestrearlo a `fs_resampled`
                            if interval_times is not None and fs_resampled is not None:
                                final_raw = data_time_to_load(raw=final_raw, intervals=interval_times, fs_resampled=fs_resampled)
                            
                            if final_raw is not None:
                                # raw_list.append(final_raw)
                                if raw_save_path is not None and type(raw_save_path) == str :
                                    final_raw.save(raw_save_path+"patient_"+str(patient)+".fif",overwrite=True) 
                                    print("________________________________________________________________________________________________________________________")
                                    print("________________________________________________________________________________________________________________________")
                                    print("________________________________________________________________________________________________________________________")
                                    print(f"File 'patient_{patient}.fif' in hour {hour} saved correctly in {raw_save_path}.")    
                                    print("________________________________________________________________________________________________________________________")
                                    print("________________________________________________________________________________________________________________________")
                                    print("________________________________________________________________________________________________________________________")
                                del raw, final_raw, raws
                            else:
                                ref_list.remove((int(patient), hour))
                                rejected.append((int(patient), hour))
                                print("________________________________________________________________________________________________________________________")
                                print("________________________________________________________________________________________________________________________")
                                print("________________________________________________________________________________________________________________________")
                                print(f"The patient {patient} in hour {hour} was rejected during preprocessing.")
                                print("________________________________________________________________________________________________________________________")
                                print("________________________________________________________________________________________________________________________")
                                print("________________________________________________________________________________________________________________________")
                            
                            # Retiro de los datos procesados que no pasaron el proceso de filtrado, referencia o epocaje, y registro de los casos rechazados
                        else:
                            ref_list.remove((int(patient), hour))
                            rejected.append((int(patient), hour))
                            print("________________________________________________________________________________________________________________________")
                            print("________________________________________________________________________________________________________________________")
                            print("________________________________________________________________________________________________________________________")
                            print(f"The patient {patient} in hour {hour} was rejected during preprocessing.")
                            print("________________________________________________________________________________________________________________________")
                            print("________________________________________________________________________________________________________________________")
                            print("________________________________________________________________________________________________________________________")
                
                else:
                    print(f"La hora {hour} no iteró en hour_list.")
            empty_recycle_bin()             
            # 5. Pipeline de Limpieza en Disco
            if remove_data:
                data_list = os.listdir(download_paht)
                print("Limpiando directorio volatil de physionet, ficheros a eliminar:")
                print(data_list)
                for data in data_list:
                    try:
                        os.remove(download_paht+data)
                        print(f"Archivo '{data}' liberado correctamente de almacenamiento.")
                    except FileNotFoundError:
                        print(f"El archivo '{data}' no existe.")
                    except OSError as e:
                        print(f"Error borrando fichero local: {e.strerror} - {e.filename}")
            
    # Retornos estructurados
    try:
        return  ref_list, rejected
    except:
        return  None, None
        


def data_time_to_load(raw: list, intervals: list[tuple[int,int]], fs_resampled: int = None) -> list:
    """
    Sub-ajusta y segmenta un bloque continuo basándose en tuplas directas orientadas al espacio del segundo en bruto.

    Parameters
    ----------
    raw : mne.io.RawArray
        El archivo de clase Raw en serie temporal sobre el cual trabajar.
    intervals : list of tuples
        Lista de tuplas [(inicio, fin), (inicio2, fin2)]. Cada elemento debe tener formato integral para delimitar un margen.
        Tenga en cuenta que el tiempo mínimo procesable debe superar el factor de recolección de 1 segundo unitario.
        Ejemplo válido: `[(0,10), (5,15), (20,30), (1200,1300)]`
    fs_resampled : int, opcional
        Nueva frecuencia fundamental si el vector resultante debe ser submuestreado espacialmente previniendo aliasing.

    Returns
    -------
    mne.io.RawArray o None
        Retorna la matriz RawArray amalgamada extraída de las marcas impuestas, o `None` en caso de que todo aborte.
    """
    if fs_resampled is None:
        signal_points = raw.copy().get_data()
        fs = raw.info["sfreq"]
    else:
        info = raw.copy().resample(fs_resampled).info
        signal_points = raw.copy().resample(fs_resampled).get_data()
        fs = fs_resampled

    if type(intervals) != list:
        print("Asegurese de inyectar las tuplas temporalmente iterativas englobadas en []")
        return "Ckeck the tuple be in a list, even if it's just one"
    
    signal_shape = signal_points.shape
    seconds_cant = int(np.floor(signal_shape[1]/fs))
    print(f"La longitud biológica de la señal está calculada fundamentalmente como {seconds_cant} segundos")
    singal_intervaled = []
    
    for interval in intervals:
        min_interval = int(interval[0]*fs) # Convertir índice temporal a coordenada del frame de la señal
        max_interval = int(interval[1]*fs)
        
        # Salvaguarda perimetral
        if min_interval > signal_shape[1] or max_interval > signal_shape[1]:
            print("-" * 5)
            print("-" * 8)
            print("-" * 10)
            print(f"Anomalía estructural en recorte de fragmentos espaciales: ({interval}). Revise duración total nativa.")
            print("-" * 10)
            print("-" * 8)
            print("-" * 5)
        else:
            print(f"El segmento index temporal {interval} (s) fue amalgamado del buffer activo a la instancia fragmentaria.")
            # Añadir rebanada del arreglo
            singal_intervaled.append(signal_points[:, min_interval:max_interval])
            
    try:
        # Concatenar todos los fragmentos temporales disconexos y unirlos
        return mne.io.RawArray(data=np.concat(singal_intervaled, axis=1), info=info)
    except:
        return None


def paching(data, window_size: int, overlapping_size: int):
    """
    Subdivide una secuencia multivariable en una lista tensorial orientada a ventanas con un nivel de empalme 'Overlap'.
    Diseñada fuertemente para generar agrupadores estáticos (batches/patches) para Aprendizaje Profundo.

    Parameters
    ----------
    data : numpy.ndarray
        Matriz original asumiendo la disposición (canales, tiempo_completo).
    window_size : int
        Tamaño base paramétrico deseado estático expresado en samples para enclochado por ventana.
    overlapping_size : int
        Cantidad de cruce traslapado en samples donde el final de T(i) coincide con la fase previa de T(i+1).

    Returns
    -------
    torch.Tensor
        Array tensorial tridimensional compatible con pytorch de dimensión (número_ventanas, canales, tamaño_ventana).
    list of tuples
        Un registro de secuencias de sub-limites documentando desde donde se trazó cada patch en su universo biológico.
    """
    patches = []
    pos = []
    start = 0
    end = window_size
    
    # Ingestar la primera rebanada base
    patches.append(data[:, start:end])
    pos.append((start, end))
    T = data.shape[1]
    
    # Correr la ventana hacia adelante de acuerdo a `overlapping_size` iterativamente
    while end < T:
        start = start + window_size - overlapping_size
        end = start + window_size
        
        # Si la próxima ventana hipotética se extruye más largo que la señal fuente natural
        if end > T:
            # Reencajar limitándola rigurosamente al máximo con solapamiento no dinámico forzado (Edge bias)
            end = T
            start = T - window_size
            pos.append((start, end))
            patches.append(data[:, start:end])
            break
            
        pos.append((start, end))
        patches.append(data[:, start:end])
    
    patches = np.stack(patches)
    # Acoplamiento directo del tensor NP al tensor TH nativo
    return from_numpy(patches), pos



def download_txt(path, save: bool, path_save: str):
    """
    Transformador de metadatos general de texto regular (`RECORDS` o `TXT`) de Physionet. Parsea información en tabla.

    Parameters
    ----------
    path : str
        Documento original separado por espacios (.txt o csv crudo) que enumera la base de datos de pacientes.
    save : bool
        Booleano de compuerta; si True volca la matriz resultante directamente al disco duro en `path_save`.
    path_save : str
        Dirección destino donde almacenar el .CSV si `save` está activo.

    Returns
    -------
    pd.DataFrame
        DataFrame estandarizado documentando jerarquías: Paciente, Horas de archivo, Extensiones (`.mat`, `.hea`, etc).
    """
    df_times = pd.read_csv(path, sep=" ", header=None)
    # Borrar headers o artefactos introductorios extraños dentro del listado maestro physionet
    df_times.drop(index=[0, 1, 162834], inplace=True)
    
    # Parseo jerárquico masivo del path crudo utilizando splits anidados
    df_times[2] = df_times[1].str.split(pat="/").apply(lambda x: x[0])
    df_times[3] = df_times[1].str.split(pat="/").apply(lambda x: x[1])
    df_times[4] = df_times[1].str.split(pat="/").apply(lambda x: x[2])
    
    # Extraer variables implícitas dentro del ID único del archivo
    df_times[5] = df_times[4].str.split(pat="_").apply(lambda x: x[1] if len(x) >= 2 else np.nan)
    df_times[6] = df_times[4].str.split(pat="_").apply(lambda x: x[2] if len(x) >= 3 else np.nan)
    df_times[7] = df_times[4].str.split(pat="_").apply(lambda x: x[3].split(sep=".")[0] if len(x) > 3 else np.nan)
    df_times[8] = df_times[4].str.split(pat=".").apply(lambda x: x[-1])
    
    # Nombrar las columnas base según el esquema jerárquico resultante de I-CARE / Physionet
    df_times.columns = ["id_path", "path", "data_partition", "patient", "file", "number_file", "hour_file", 
                 "format_data", "file_extension"]
                 
    # Inyectar campos semánticos vacíos para rellenar post-procesado posteriormente tras descarga
    df_times["meas_date"] = pd.Series()
    df_times["minutes"] = pd.Series()
    df_times["#_channels"] = pd.Series()
    df_times["channels"] = pd.Series()
    df_times["fs"] = pd.Series()
    df_times["time_points"] = pd.Series()
    
    if save:
        df_times.to_csv(path_or_buf=path_save, sep=";", index=False)
    return df_times


def interpolation_corr(raw, ch_name):
    """
    Fuerza heurísticamente el silenciamiento de un canal válido conocido, lo interpola pseudo-topográficamente
    y diagnostica su robustez calculando la correlación R de Pearson al original.

    Parameters
    ----------
    raw : mne.io.RawArray
        Datos crudos de MNE origen (necesariamente debe traer los montajes de layout standard referenciados).
    ch_name : str
        Vector ID de string a silenciar / aislar temporalmente para prueba.

    Returns
    -------
    float
        Valor 'r' métrico (De -1 a 1 indicando correlación Pearson de la estimación matemática vs. la biológica real).
    """
    raw_copy = raw.copy()
    
    # Extraer la métrica base origen antes de silenciarla (Suelo Verdadero / Ground Truth)
    x_orig = raw_copy.get_data(picks=[ch_name])[0]

    # Etiquetar artificialmente fallida y mandar a interpolar esféricamente (Spherical splines fallback)
    raw_copy.info["bads"] = [ch_name]
    raw_copy.interpolate_bads(reset_bads=True)
    
    # Consolidar la variante sintética derivada de datos vecinos y comparar
    x_interp = raw_copy.get_data(picks=[ch_name])[0]

    r, _ = pearsonr(x_orig, x_interp)
    return r




def read_raws(path_files: str, time_taken: float = None, del_files=False):
    """
    Recorredor nativo de almacenamiento en bruto. Levanta al vuelo cualquier pre-procesado listado
    `.fif` (binario temporal en crudo generalizado por MNE) del disco y valida longitudes biológicas mínimas de la serie.

    Parameters
    ----------
    path_files : str
        Entrada hacia el sistema de archivos del SO en forma de cadena estandarizada local.
    time_taken : float, opcional
        Parámetro de barrera. El tiempo estrictamente demandado en segundos base a aplicar rigurosamente en `del_files`.
    del_files : bool, opcional
        Determina si el motor purgará / ignorará por completo recortes o pacientes que quedaron demasiado pequeños 
        (ej: `shape < time_taken * fs`).

    Returns
    -------
    tuple
        Retorna (`raw_list`, `patients`). Lista de instancias válidas MNE reconstruidas y lista de IDs crudos.
    """
    file_list = os.listdir(path_files)
    
    # Carga Pre-Launch de MNE asegurándose que la RAM los acople sin retrasos lazy (preload=True)
    raw_list = [mne.io.read_raw_fif(path_files+file, preload=True) for file in file_list]
    
    # Suponer convencionalmente que el nombre del fichero FIF es `xxx_1234.fif`
    patients = [file.split(".")[0][-4:] for file in file_list]
    
    # Profiler automático: Validar anomalías de truncamiento o falta biológica frente a un target implícito
    for i in range(len(raw_list)):
        if raw_list[i].get_data().shape[1] < 40 * 128:  # Advierte si una grabación duró intencionalmente menos
            print(f"Paciente con longitud atípicamente precaria detectado en índice: {i}, Longitud registrada: {raw_list[i].get_data().shape[1]}")
            print(f"ID del Paciente atípico: {patients[i]}")
    
    # Cortar e inutilizar ficheros de tamaño irregular descartando del Output
    if del_files:
        freq = raw_list[0].info["sfreq"]
        
        # Filtrado vectorial basado en la longitud canónica dictada por la Frecuencia * Segundos
        valid_indices = [i for i in range(len(raw_list)) if raw_list[i].get_data().shape[1] == time_taken * freq]
        raw_list = [raw_list[i] for i in valid_indices]
        patients = [patients[i] for i in valid_indices]
    
    return raw_list, patients