"""
Bollettino giornaliero con sistema di First Guess da ICON 2I, aggiornamento giornaliero per i RUN 00 e 12

@author: deniel
"""

#  VERSIONE CON KRIGING E CORREZIONE ALTITUDINE

import os
import requests
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo
import pickle

# === CONFIG ===
DATA_DIR = os.path.join(os.getcwd(), "data")
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
RUN_HOURS = ['00', '12']
VARIABLES = ['T_2M', 'RELHUM', 'TOT_PREC', 'CLCT', 'PMSL', 'U_10M', 'V_10M', 'HSURF']
CAPOLUOGHI = {
    'Agrigento': (37.3111, 13.5765, 230),
    'Alessandria': (44.9129, 8.6153, 95),
    'Ancona': (43.6158, 13.5189, 16),
    'Aosta': (45.7370, 7.3201, 583),
    'Arezzo': (43.4633, 11.8799, 296),
    'Ascoli Piceno': (42.8530, 13.5768, 154),
    'Asti': (44.9007, 8.2064, 123),
    'Avellino': (40.9140, 14.7951, 348),
    'Bari': (41.1171, 16.8719, 5),
    'Barletta': (41.3190, 16.2810, 15),
    'Andria': (41.2310, 16.2900, 151),
    'Trani': (41.2750, 16.4160, 7),
    'Belluno': (46.1410, 12.2170, 390),
    'Benevento': (41.1298, 14.7819, 135),
    'Bergamo': (45.6983, 9.6773, 249),
    'Biella': (45.5616, 8.0588, 420),
    'Bologna': (44.4949, 11.3426, 54),
    'Bolzano': (46.4983, 11.3548, 262),
    'Brescia': (45.5416, 10.2118, 149),
    'Brindisi': (40.6320, 17.9361, 10),
    'Cagliari': (39.2238, 9.1217, 6),
    'Caltanissetta': (37.4900, 14.0627, 568),
    'Campobasso': (41.5600, 14.6600, 701),
    'Caserta': (41.0747, 14.3323, 68),
    'Catania': (37.5079, 15.0830, 7),
    'Catanzaro': (38.8896, 16.6052, 320),
    'Chieti': (42.3510, 14.1675, 330),
    'Como': (45.8080, 9.0852, 201),
    'Cosenza': (39.2990, 16.2534, 238),
    'Cremona': (45.1333, 10.0227, 45),
    'Crotone': (39.0833, 17.1167, 8),
    'Cuneo': (44.3894, 7.5486, 534),
    'Enna': (37.5670, 14.2794, 931),
    'Fermo': (43.1629, 13.7196, 319),
    'Ferrara': (44.8354, 11.6199, 9),
    'Firenze': (43.7696, 11.2558, 50),
    'Foggia': (41.4622, 15.5446, 76),
    'Forlì': (44.2227, 12.0407, 34),
    'Cesena': (44.1390, 12.2430, 44),
    'Frosinone': (41.6398, 13.3519, 291),
    'Genova': (44.4056, 8.9463, 19),
    'Gorizia': (45.9412, 13.6214, 84),
    'Grosseto': (42.7631, 11.1122, 10),
    'Imperia': (43.8896, 8.0386, 10),
    'Isernia': (41.5910, 14.2326, 423),
    'L\'Aquila': (42.3498, 13.3995, 714),
    'La Spezia': (44.1072, 9.8289, 10),
    'Latina': (41.4676, 12.9037, 21),
    'Lecce': (40.3529, 18.1743, 49),
    'Lecco': (45.8566, 9.3974, 214),
    'Livorno': (43.5485, 10.3106, 3),
    'Lodi': (45.3142, 9.5037, 87),
    'Lucca': (43.8376, 10.4951, 19),
    'Macerata': (43.3003, 13.4530, 315),
    'Mantova': (45.1564, 10.7914, 19),
    'Massa-Carrara': (44.0340, 10.1390, 65),
    'Matera': (40.6663, 16.6043, 401),
    'Messina': (38.1938, 15.5540, 3),
    'Milano': (45.4642, 9.1900, 120),
    'Modena': (44.6471, 10.9252, 34),
    'Monza e Brianza': (45.5845, 9.2744, 162),
    'Napoli': (40.8518, 14.2681, 17),
    'Novara': (45.4450, 8.6222, 162),
    'Nuoro': (40.3212, 9.3298, 554),
    'Oristano': (39.9043, 8.5923, 10),
    'Padova': (45.4064, 11.8768, 12),
    'Palermo': (38.1157, 13.3615, 14),
    'Parma': (44.8015, 10.3279, 57),
    'Pavia': (45.1847, 9.1582, 77),
    'Perugia': (43.1122, 12.3888, 493),
    'Pesaro e Urbino': (43.9090, 12.9159, 11),  # media indicativa
    'Pescara': (42.4643, 14.2134, 4),
    'Piacenza': (45.0526, 9.6927, 61),
    'Pisa': (43.7162, 10.3966, 4),
    'Pistoia': (43.9335, 10.9180, 67),
    'Pordenone': (45.9569, 12.6605, 24),
    'Potenza': (40.6401, 15.8050, 819),
    'Prato': (43.8777, 11.1022, 65),
    'Ragusa': (36.9256, 14.7245, 502),
    'Ravenna': (44.4184, 12.2035, 4),
    'Reggio Calabria': (38.1105, 15.6613, 31),
    'Reggio Emilia': (44.6983, 10.6301, 58),
    'Rieti': (42.4048, 12.8608, 405),
    'Rimini': (44.0604, 12.5653, 6),
    'Roma': (41.9028, 12.4964, 21),
    'Rovigo': (45.0706, 11.7905, 7),
    'Salerno': (40.6824, 14.7681, 4),
    'Sassari': (40.7259, 8.5555, 225),
    'Savona': (44.3090, 8.4772, 4),
    'Siena': (43.3188, 11.3308, 322),
    'Siracusa': (37.0755, 15.2866, 17),
    'Sondrio': (46.1690, 9.8710, 307),
    'Sud Sardegna': (39.1672, 8.5150, 70),  # media indicativa
    'Taranto': (40.4736, 17.2429, 15),
    'Teramo': (42.6612, 13.6988, 265),
    'Terni': (42.5636, 12.6439, 130),
    'Torino': (45.0703, 7.6869, 239),
    'Trapani': (38.0176, 12.5360, 3),
    'Trento': (46.0700, 11.1200, 194),
    'Treviso': (45.6669, 12.2431, 15),
    'Trieste': (45.6495, 13.7768, 2),
    'Udine': (46.0626, 13.2345, 113),
    'Varese': (45.8206, 8.8255, 382),
    'Venezia': (45.4408, 12.3155, 2),
    'Verbano-Cusio-Ossola': (45.9211, 8.5518, 212),
    'Vercelli': (45.3239, 8.4196, 130),
    'Verona': (45.4384, 10.9916, 59),
    'Vibo Valentia': (38.6762, 16.1000, 476),
    'Vicenza': (45.5469, 11.5475, 39),
    'Viterbo': (42.4207, 12.1077, 326),

    # Montagna (Nord)
    'Cortina d\'Ampezzo': (46.5405, 12.1357, 1224),
    'Madonna di Campiglio': (46.2296, 10.8266, 1550),
    'Livigno': (46.5382, 10.1413, 1816), 
    'Courmayeur': (45.7918, 6.9650, 1224),
    'Sestriere': (44.9570, 6.8797, 2035),
    'Blinnenhorn': (46.424794, 8.308183, 3374),
    'Formazza': (46.372819, 8.426935, 1380),
    "Bivacco Alpe Mottac - 1690 m": (46.061138, 8.404305, 1690),

    # Mare (Nord e Centro)
    'Termoli': (42.0036, 14.9981, 15), 
    'Rimini': (44.0604, 12.5653, 5),
    'Viareggio': (43.8662, 10.2485, 2),
    'Sanremo': (43.8170, 7.7760, 15),
    'Cefalù': (38.0380, 14.0186, 16),
    'Taormina': (37.8516, 15.2852, 204),

    # Mare (Sud)
    'Gallipoli': (40.0555, 17.9975, 12),
    'Polignano a Mare': (40.9950, 17.2185, 24),
    'Tropea': (38.6762, 15.8982, 61),
    'Maratea': (39.9886, 15.7136, 300),
    'Capri': (40.5532, 14.2220, 142),
    'Vieste': (41.881510, 16.171112, 43),

    # Isole (piccole)
    'Lampedusa': (35.5022, 12.6185, 16),
    'Ischia': (40.7376, 13.9481, 90),
    'Elba': (42.8142, 10.3160, 344),
    'Pantelleria': (36.8335, 11.9474, 836),
    'La Maddalena': (41.2131, 9.4054, 19),
    
    # Mare
    'Mar_Tirreno': (39.753155, 12.0000, 0),
    'Mar_Adriatico': (42.974424, 15.278642, 0),
    'Mar_Ionio': (37.965580, 17.944965, 0),
    
    'Montesilvano': (42.509602, 14.142485, 5),
    'Campo Imperatore': (42.442811, 13.558681, 2130),
    'Ristoro Mucciante': (42.407376, 13.744723, 1800),
    'Ghiacciaio Calderone': (42.471596, 13.568656, 2871),
    
    'Casteldelci (RN) - 632 m': (43.791336, 12.154737, 632),
    'Tuscania (VT) - 165 m': (42.419884, 11.869364, 165),
    'Bolsena (VT) - 350 m': (42.644598, 11.986747, 350),
    "Scanno (AQ) - 1050 m": (41.903743, 13.880701, 1050),
    "Caramanico Terme (PE) - 650 m": (42.157698, 14.002185, 650),
    "Castelnuovo Magra (SP) - 190 m": (44.099743, 10.017324, 190),
    "Faenza": (44.2854, 11.8833, 35),
    "Mirabella Eclano": (41.0573, 14.9931, 372),
    "Verona": (45.4330, 10.9830, 62),
    "Padova": (45.4080, 11.8840, 11),
    'Nulvi': (40.7962, 8.7465, 478),
    'Montecosaro': (43.3083, 13.6269, 252),
    'Sutera': (37.5090, 13.7300, 590),
    "Pescasseroli": (41.8034, 13.7871, 1167)
}

# === UTILS ===

def kelvin_to_celsius(k): return k - 273.15
def wind_speed_direction(u, v): return np.sqrt(u**2 + v**2) * 3.6, (np.arctan2(-u, -v) * 180 / np.pi) % 360
def wind_dir_to_cardinal(deg):
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    return dirs[int((deg + 22.5) % 360 // 45)]

def download_icon_data():
    now = datetime.utcnow()
    if now.hour < 3:
        run_hour = '12'; run_date = (now - timedelta(days=1)).strftime('%Y%m%d')
    elif now.hour < 14:
        run_hour = '00'; run_date = now.strftime('%Y%m%d')
    else:
        run_hour = '12'; run_date = now.strftime('%Y%m%d')

    base_url = f'https://meteohub.mistralportal.it/nwp/ICON-2I_SURFACE_PRESSURE_LEVELS/{run_date}{run_hour}/'
    os.makedirs(DATA_DIR, exist_ok=True)

    for var in VARIABLES:
        var_url = f'{base_url}{var}/'
        try:
            r = requests.get(var_url)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            grib_files = [a.get('href') for a in soup.find_all('a') if a.get('href', '').endswith('.grib')]
            if not grib_files: continue
            file_url = var_url + grib_files[0]
            local_path = os.path.join(DATA_DIR, f'{var}.grib')

            skip = False
            if os.path.exists(local_path):
                head = requests.head(file_url)
                if 'Last-Modified' in head.headers:
                    remote = datetime.strptime(head.headers['Last-Modified'], '%a, %d %b %Y %H:%M:%S %Z')
                    local = datetime.utcfromtimestamp(os.path.getmtime(local_path))
                    if local >= remote:
                        print(f'{var} già  aggiornato.'); skip = True

            if not skip:
                with requests.get(file_url, stream=True) as resp:
                    with open(local_path, 'wb') as f: f.write(resp.content)
                print(f'Scaricato {var}')
        except Exception as e:
            print(f'Errore download {var}: {e}')

def load_data():
    data = {}
    for var in VARIABLES:
        path = os.path.join(DATA_DIR, f'{var}.grib')
        if os.path.exists(path):
            try:
                ds = xr.open_dataset(path, engine='cfgrib')
                data[var] = ds
            except Exception as e:
                print(f'Errore lettura {var}: {e}')
    return data

def extract_variable(var, lat_idx, lon_idx):
    """
    Estrae media pesata su 1 centro + 8 vicini + 16 anello esterno.
    """
    if var.ndim == 3:  # (time, lat, lon)
        center = var[:, lat_idx, lon_idx]
        first_ring = [var[:, lat_idx + i, lon_idx + j]
                      for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)]
        second_ring = [var[:, lat_idx + i, lon_idx + j]
                       for i in [-2, -1, 0, 1, 2]
                       for j in [-2, -1, 0, 1, 2]
                       if (abs(i) == 2 or abs(j) == 2) and (0 <= lat_idx + i < var.shape[1]) and (0 <= lon_idx + j < var.shape[2])]

        first_ring = np.stack(first_ring, axis=0).mean(axis=0)
        second_ring = np.stack(second_ring, axis=0).mean(axis=0) if second_ring else 0

        return 0.5 * center + 0.25 * first_ring + 0.25 * second_ring

    elif var.ndim == 2:  # (lat, lon)
        return var[lat_idx, lon_idx]

    else:
        raise ValueError(f"Array con shape non gestita: {var.shape}")

def wet_bulb_temperature(t2m, rh2m):
    RH = np.clip(rh2m, 0.1, 100)
    return (t2m * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) +
            np.arctan(t2m + RH) -
            np.arctan(RH - 1.676331) +
            0.00391838 * RH**1.5 * np.arctan(0.023101 * RH) -
            4.686035)

def weather_data(data):
    t2m_raw = kelvin_to_celsius(data['T_2M']['t2m'].values)
    rh2m_raw = data['RELHUM']['r'].values
    tp_raw = np.diff(data['TOT_PREC']['tp'].values, axis=0, prepend=0)
    clct_raw = data['CLCT']['clct'].values
    pmsl_raw = data['PMSL']['pmsl'].values / 100
    u10_raw = data['U_10M']['u10'].values
    v10_raw = data['V_10M']['v10'].values
    hsurf_raw = data['HSURF']['hsurf'].values

    wind_speed, wind_deg = wind_speed_direction(u10_raw, v10_raw)
    wind_card = np.vectorize(wind_dir_to_cardinal)(wind_deg)

    nlat, nlon = hsurf_raw.shape
    lats = np.linspace(33.7, 48.89, nlat)
    lons = np.linspace(3, 22, nlon)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    def find_nearest(lat, lon):
        dist = np.sqrt((lat_grid - lat)**2 + (lon_grid - lon)**2)
        return np.unravel_index(np.argmin(dist), dist.shape)

    capoluoghi_dati = {}
    for city, (lat_c, lon_c, alt_c) in CAPOLUOGHI.items():
        lat_idx, lon_idx = find_nearest(lat_c, lon_c)

        hs_loc = extract_variable(hsurf_raw, lat_idx, lon_idx)
        delta_z = -alt_c + hs_loc

        lapse_rate = 0.0065  # °C/m
        p_lapse_rate = 12  # hPa ogni 100 m

        t2m = extract_variable(t2m_raw, lat_idx, lon_idx) + lapse_rate * delta_z
        rh2m = np.clip(extract_variable(rh2m_raw, lat_idx, lon_idx), 0, 100)
        tp = extract_variable(tp_raw, lat_idx, lon_idx)
        clct = extract_variable(clct_raw, lat_idx, lon_idx)
        pmsl = extract_variable(pmsl_raw, lat_idx, lon_idx) + (delta_z / 100) * p_lapse_rate

        u10 = extract_variable(u10_raw, lat_idx, lon_idx)
        v10 = extract_variable(v10_raw, lat_idx, lon_idx)
        ws, wd_deg = wind_speed_direction(u10, v10)
        wd_card = np.vectorize(wind_dir_to_cardinal)(wd_deg)

        capoluoghi_dati[city] = {
            't2m': np.round(t2m, 1),
            'rh2m': np.round(rh2m, 1),
            'tp': np.round(tp, 1),
            'clct': np.round(clct, 1),
            'pmsl': np.round(pmsl, 1),
            'wind_speed': np.round(ws, 1),
            'wind_dir_cardinal': wd_card,
            'tw': np.round(wet_bulb_temperature(t2m, rh2m), 1),
            'lat': lat_c,
            'lon': lon_c,
            'alt_model': round(hs_loc, 1),
            'alt_real': alt_c
        }
    return capoluoghi_dati

# === MAIN ===
if __name__ == '__main__':
    now = datetime.utcnow()
    if now.hour < 3:
        run_hour = '12'; run_date = (now - timedelta(days=1)).strftime('%Y%m%d')
    elif now.hour < 14:
        run_hour = '00'; run_date = now.strftime('%Y%m%d')
    else:
        run_hour = '12'; run_date = now.strftime('%Y%m%d')

    run_datetime_utc = datetime.strptime(run_date + run_hour, '%Y%m%d%H').replace(tzinfo=ZoneInfo('UTC'))
    download_icon_data()
    data = load_data()
    if data:
        capoluoghi_dati = weather_data(data)
        pickle_path = os.path.join(DATA_DIR, 'capoluoghi_dati.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump({'capoluoghi_dati': capoluoghi_dati, 'run_datetime_utc': run_datetime_utc}, f)
        print(f'Dati salvati in {pickle_path}')
        
# %%   BOLLETTINO GIORNALIERO

from datetime import timedelta
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.image import imread
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import rasterio
import shapely.geometry as sgeom
import cartopy.io.shapereader as shpreader
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.colors import to_rgba
from shapely.ops import unary_union
import pickle
from collections import Counter
import pytz
import locale

print("Inizia creazione bollettino nazionale giornaliero...")
# Directory
data_dir = os.path.join(os.getcwd(), "data")
print(data_dir)
icone_dir = os.path.join(os.getcwd(), "icons")
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

# Lista dei capoluoghi di regione
capoluoghi_regione = {
    'Torino', 'Aosta', 'Milano', 'Trento', 'Venezia', 'Trieste', 'Genova', 'Bologna', 'Firenze', 'Perugia',
    'Ancona', 'Roma', 'L\'Aquila', 'Campobasso', 'Napoli', 'Potenza', 'Bari', 'Catanzaro', 'Palermo', 'Cagliari'
}
# Aggiunta delle localita extra
localita_extra = {
    "Bolzano", 'Sassari', 'Catania', 'Lecce', 'Elba', 'Foggia', 'Livigno',
    'Mar_Ionio', 'Mar_Tirreno', 'Mar_Adriatico', 'Formazza', 'Pantelleria', "Cortina d\'Ampezzo", "Parma"
}
# Unione degli insiemi
localita_interessate = capoluoghi_regione.union(localita_extra)

# Costante zona oraria italiana
tz_italy = ZoneInfo('Europe/Rome')

# Sfondo mappa
satellite_path = os.path.join(icone_dir, "satellite.tif")

pickle_path = os.path.join(data_dir, 'capoluoghi_dati.pkl')
print(pickle_path)
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)
capoluoghi_dati = data['capoluoghi_dati']
run_datetime_utc = data['run_datetime_utc']


# Crea nome cartella tipo '20250524'
run_folder = run_datetime_utc.strftime('%Y%m%d')
# Percorso completo di output
run_output_dir = os.path.join(output_dir, run_folder)
# Crea la cartella se non esiste
os.makedirs(run_output_dir, exist_ok=True)

# Offsets personalizzati per le città  (in gradi lat/lon)
offset_icone = {
    'Bologna': (0.2, 0.05),  # sposta Bologna 0.3° più a nord (lat), nessun cambio in lon
    'Firenze': (-0.1,0),
    'Genova': (0,0.05),
    'Torino': (-0.25, 0.1),
    "Bolzano": (0.1,0.1),
    'Livigno': (0.1,-0.1),
    'Campobasso': (0.15,0.05),
    'Mare Adriatico': (0,0.4),
    'Elba': (0,-0.1),
    'Trieste': (0,-0.1),
    'Foggia': (0.25, 0.15),
    'L\'Aquila': (0,0.05),
    'Bolzano': (0.2,0.2),
    'Cortina d\'Ampezzo': (0.25,0.35),
    'Trento': (-0.25,-0.05),
    'Parma': (-0.05,0)
    # aggiungi altre città  se necessario
}


# Funzione per determinare l'icona meteo
def icona_meteo(clct, tp, tw, ora_locale, wind_speed, nome_città ):

    # Aggiunta mare (opzionale, se rilevante)
    if nome_città .startswith("Mar_"):
        if wind_speed < 10:
            return f"mare_1.png"
        elif wind_speed < 20:
            return f"mare_2.png"
        else:
            return f"mare_3.png"
    else:
        giorno_anno = ora_locale.timetuple().tm_yday
        if 80 <= giorno_anno <= 265:
            alba, tramonto = 5, 20
        elif 20 <= giorno_anno < 80:
            alba, tramonto = 8, 17
        else:
            alba, tramonto = 7, 18
            
        ora_decimale = 12  # Forza giorno
        giorno = alba <= ora_decimale <= tramonto
        suffisso = "_g" if giorno else "_n"

        if clct < 20:
            cielo = "sereno"
        elif clct < 50:
            cielo = "poconuv"
        elif clct < 80:
            cielo = "nuv"
        else:
            cielo = "cop"

        if tp < 0.4:
            return f"{cielo}{suffisso}.png"
        else:
            if tw <= 0.5:
                if tp < 5:
                    return f"{cielo}_1n{suffisso}.png"
                elif tp < 30:
                    return f"{cielo}_2n{suffisso}.png"
                else:
                    return f"{cielo}_3n{suffisso}.png"
            else:
                if tp < 5:
                    return f"{cielo}_1p{suffisso}.png"
                elif tp < 30:
                    return f"{cielo}_2p{suffisso}.png"
                else:
                    return f"{cielo}_3p{suffisso}.png"

# Funzione per schiarire tutto fuori Italia
def schiarisci_fuori_italia(ax):
    extent = ax.get_extent(crs=ccrs.PlateCarree())
    minx, maxx, miny, maxy = extent
    outer = sgeom.box(minx, miny, maxx, maxy)

    shapename = 'admin_0_countries'
    countries_shp = shpreader.natural_earth(resolution='10m', category='cultural', name=shapename)
    reader = shpreader.Reader(countries_shp)

    italy_geom = None
    all_land = []

    for record in reader.records():
        geom = record.geometry
        all_land.append(geom)
        if record.attributes['NAME_EN'] == 'Italy':
            italy_geom = geom

    if italy_geom is None:
        print("Errore: geometria Italia non trovata!")
        return

    # Unisci tutta la terraferma (unione di tutti i paesi)
    all_land_union = unary_union(all_land)

    # Calcola la parte di terraferma fuori Italia
    mask_geom = all_land_union.difference(italy_geom)

    # Converti in patch matplotlib (stessa funzione di prima)
    def shapely_to_pathpatch(shapely_geom, **kwargs):
        if shapely_geom.geom_type == 'Polygon':
            verts = []
            codes = []

            exterior = shapely_geom.exterior
            verts += list(exterior.coords)
            codes += [Path.MOVETO] + [Path.LINETO]*(len(exterior.coords)-2) + [Path.CLOSEPOLY]

            for interior in shapely_geom.interiors:
                verts += list(interior.coords)
                codes += [Path.MOVETO] + [Path.LINETO]*(len(interior.coords)-2) + [Path.CLOSEPOLY]

            path = Path(verts, codes)
            patch = PathPatch(path, **kwargs)
            return patch
        elif shapely_geom.geom_type == 'MultiPolygon':
            patches = []
            for part in shapely_geom.geoms:
                patches.append(shapely_to_pathpatch(part, **kwargs))
            return patches
        else:
            return None

    patch = shapely_to_pathpatch(mask_geom, facecolor=to_rgba('white', 0.5), edgecolor='none', transform=ccrs.PlateCarree())
    
    if isinstance(patch, list):
        for p in patch:
            ax.add_patch(p)
    else:
        ax.add_patch(patch)


# # Numero di step temporali (devi definire capoluoghi_dati e run_datetime_utc prima)
# n_time = capoluoghi_dati[next(iter(capoluoghi_dati))]['t2m'].shape[0]
# n_days = n_time // 24

# for d in range(n_days):
    # start_utc = run_datetime_utc + timedelta(days=d)
    # start_loc = start_utc.astimezone(tz_italy)

# Determina se il run è alle 12 UTC
run_hour = run_datetime_utc.hour

# Calcola il numero totale di giorni disponibili nei dati
n_time = capoluoghi_dati[next(iter(capoluoghi_dati))]['t2m'].shape[0]
n_days = n_time // 24

# Determina il giorno iniziale per la generazione dei bollettini
start_day = 0
start_idx_offset = 0
offset_cet = 0
if run_hour == 12:
    start_day = 1  # Salta il primo giorno se il run è alle 12 UTC
    start_idx_offset = -12
    # Calcolo offset ora legale/solare
    tz_italy = pytz.timezone("Europe/Rome")
    now = datetime.now(tz_italy)
    offset_cet = int(now.utcoffset().total_seconds() // 3600)

for d in range(start_day, n_days):
    # Calcola l'indice di inizio e fine per i dati del giorno corrente
    start_idx = d * 24 + start_idx_offset + offset_cet
    end_idx = (d + 1) * 24 + start_idx_offset + offset_cet

    # Calcola la data del bollettino
    start_utc = run_datetime_utc + timedelta(days=d)
    start_loc = start_utc.astimezone(tz_italy)
    fig = plt.figure(figsize=(12, 12))
    fig.patch.set_facecolor('#157acc')
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([5.75, 19.25, 35.5, 47.5], crs=ccrs.PlateCarree())

    with rasterio.open(satellite_path) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        ax.imshow(img, origin='upper', extent=extent, transform=ccrs.PlateCarree())

    # Schiarisci fuori Italia
    schiarisci_fuori_italia(ax)

    ax.coastlines(resolution='10m', zorder=0)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), zorder=0)

    # Filtraggio localita durante il ciclo
    for city, dati in capoluoghi_dati.items():
        if city not in localita_interessate:
            continue  # salta città  non rilevanti

        lat, lon = dati['lat'], dati['lon']
        # Applica offset personalizzato se presente
        offset_lat, offset_lon = offset_icone.get(city, (0, 0))
        lat = lat + offset_lat
        lon = lon + offset_lon
        
        t2m = dati['t2m'][start_idx:end_idx]
        t2m_min = np.min(t2m)
        t2m_max = np.max(t2m)
        rh2m_avg = np.mean(dati['rh2m'][start_idx:end_idx])
        tp_sum = np.sum(dati['tp'][start_idx:end_idx])
        clct_avg = np.mean(dati['clct'][start_idx:end_idx])
        pmsl_avg = np.mean(dati['pmsl'][start_idx:end_idx])
        wind_speed_avg = np.mean(dati['wind_speed'][start_idx:end_idx])
        wind_dir_avg = Counter(dati['wind_dir_cardinal'][start_idx:end_idx]).most_common(1)[0][0]
        tw_avg = np.mean(dati['tw'][start_idx:end_idx])

        nome_icona = icona_meteo(clct_avg, tp_sum, tw_avg, start_loc, wind_speed_avg, city)
        path_icona = os.path.join(icone_dir, nome_icona)

        if os.path.exists(path_icona):
            img_icon = imread(path_icona)
            ab = AnnotationBbox(OffsetImage(img_icon, zoom=0.035), (lon, lat), frameon=False, transform=ccrs.PlateCarree())
            ax.add_artist(ab)
        else:
            print(f"Icona mancante: {nome_icona}")

        # Riquadro bianco trasparente più in basso e centrato
        

        if city.startswith("Mar_"):
            
            box_x = lon - 0.35
            box_y = lat - 0.35
            
            text_y = box_y + 0.075
            
            # ax.add_patch(plt.Rectangle(
            #     (box_x, box_y), 0.8, 0.15,
            #     transform=ccrs.PlateCarree(),
            #     color='white', alpha=0.7, zorder=10
            # ))

            # # Offset per posizionare le due parti del testo correttamente
            # dir_x = box_x + 0.025
            # rest_x = box_x + 0.225  # distanza dopo la parte bold
            # wind_text_rest = f" - {int(wind_speed_avg)} km/h"

            # # Parte in grassetto (direzione)
            # ax.text(dir_x, text_y, wind_dir_avg,
            #         color='black', ha='left', va='center',
            #         fontsize=6, weight='bold',
            #         transform=ccrs.PlateCarree(), zorder=11)

            # # Parte normale (velocità  vento)
            # ax.text(rest_x, text_y, wind_text_rest,
            #         color='black', ha='left', va='center',
            #         fontsize=6, weight='normal',
            #         transform=ccrs.PlateCarree(), zorder=11)

        else:
            
            box_x = lon - 0.3
            box_y = lat - 0.5
            
            text_y = box_y + 0.075
            
            ax.add_patch(plt.Rectangle(
                (box_x, box_y), 0.7, 0.15,
                transform=ccrs.PlateCarree(),
                color='white', alpha=0.7, zorder=10
            ))
            
            # Temperatura min/max
            ax.text(box_x + 0.20, text_y, f"{round(t2m_min)}", color='tab:blue', ha='right', va='center',
                    fontsize=6, weight='bold', transform=ccrs.PlateCarree(), zorder=11)
            ax.text(box_x + 0.25, text_y, "/", color='black', ha='center', va='center',
                    fontsize=6, transform=ccrs.PlateCarree(), zorder=11)
            ax.text(box_x + 0.30, text_y, f"{round(t2m_max)}", color='tab:red', ha='left', va='center',
                    fontsize=6, weight='bold', transform=ccrs.PlateCarree(), zorder=11)
            ax.text(box_x + 0.45, text_y, "°C", color='black', ha='left', va='center',
                    fontsize=6, transform=ccrs.PlateCarree(), zorder=11)

    logo_path = os.path.join(icone_dir, "image001.png")
    if os.path.exists(logo_path):
        logo_img = imread(logo_path)
        imagebox = OffsetImage(logo_img, zoom=0.07)  # regola zoom per stare accanto al titolo
        ab_logo = AnnotationBbox(imagebox, (0.18, 0.05), xycoords=ax.transAxes,
                                 frameon=False, box_alignment=(1, 1), zorder=15)
        ax.add_artist(ab_logo)
    else:
        print("Logo non trovato:", logo_path)
    run_hour = run_datetime_utc.strftime('%H')
    # Imposta la lingua italiana per i nomi dei giorni
    locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')  # Su Windows potresti dover usare 'italian'
    # Supponiamo che start_loc sia un datetime
    line1 = f"Bollettino Italia - {start_loc.strftime('%A %d/%m/%Y')}"
    line2 = f"ICON 2I - run: {run_datetime_utc.strftime('%d/%m/%Y %H')}"
    # x=0.5 centro, y=1.02 e y=0.98 per due righe ravvicinate sopra l'area della mappa
    ax.text(0.5, 1.05, line1, ha='center', va='bottom', fontsize=16, weight='bold', transform=ax.transAxes)
    ax.text(0.5, 1.03, line2, ha='center', va='top', fontsize=12, transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(f"{run_output_dir}/{start_loc.strftime('%d-%m-%Y')}_{run_hour}.png", dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print("Fine creazione bollettino giornaliero.")
    
    
# %%   BOLLETTINO TRIORARIO PDF PER LOCALITÀ

import os
import pickle
import numpy as np
import pandas as pd
from datetime import timedelta, datetime, timezone
from fpdf import FPDF
from PIL import Image
import locale

print("Inizio creazione bollettino triorario per località...")
ICONS_PATH = os.path.join(os.getcwd(), "icons")
FONT_PATH = os.path.join(ICONS_PATH, "DejaVuSans.ttf")
LOGO_PATH = os.path.join(ICONS_PATH, "image001.png")

ICON_CACHE = {}

# Costanti per la Logica Meteo
SUMMER_START_DAY = 80    # ~21 Marzo
SUMMER_END_DAY = 265     # ~22 Settembre
ALBA_SUMMER = 5
TRAMONTO_SUMMER = 20
ALBA_WINTER_EARLY = 8
TRAMONTO_WINTER_EARLY = 17
ALBA_WINTER_LATE = 7
TRAMONTO_WINTER_LATE = 18

CLOUD_CLEAR_THRESHOLD = 20
CLOUD_PARTLY_THRESHOLD = 50
CLOUD_MOSTLY_THRESHOLD = 80

RAIN_LIGHT_THRESHOLD = 0.3
RAIN_MEDIUM_THRESHOLD = 5
RAIN_HEAVY_THRESHOLD = 30

WIND_MARE_CALM_THRESHOLD = 10
WIND_MARE_MODERATE_THRESHOLD = 20

NUM_HOURS_FORECAST = 72
GROUPING_INTERVAL_HOURS = 3 # Intervallo di raggruppamento (3 ore per triorario)

# Costanti per lo Stile PDF
HEADER_FONT_SIZE = 16
TABLE_TITLE_FONT_SIZE = 14
NORMAL_FONT_SIZE = 10
SMALL_FONT_SIZE = 7

COLOR_HEADER_BG = (200, 220, 240)
COLOR_ROW_EVEN_BG = (240, 240, 240)
COLOR_ROW_ODD_BG = (255, 255, 255)
COLOR_TEXT = (0, 0, 0)
COLOR_BORDER = (180, 180, 180)
BORDER_LINE_WIDTH = 0.2

ICON_DISPLAY_WIDTH = 8
ICON_DISPLAY_HEIGHT = 8
CELL_HEIGHT = 12

COLOR_TEMP_MIN = (0, 0, 255) # Blu
COLOR_TEMP_MAX = (255, 0, 0) # Rosso
COLOR_PRECIP = (0, 0, 0)     # Nero

# Funzioni Ausiliarie

def get_cached_icon_path(icon_filename):
    """
    Restituisce il percorso completo dell'icona e la mette in cache se non già  presente.
    """
    if icon_filename not in ICON_CACHE:
        full_path = os.path.join(ICONS_PATH, icon_filename)
        if not os.path.exists(full_path):
            print(f"Attenzione: Icona non trovata: {full_path}")
            return None
        ICON_CACHE[icon_filename] = full_path
    return ICON_CACHE[icon_filename]

def get_weather_icon_filename(clct, tp, t2m, ora_locale, wind_speed, nome_citta):
    """
    Determina il nome del file dell'icona meteo in base alle condizioni, senza bulbo umido.
    """
    if nome_citta.startswith("Mar_"):
        if wind_speed < WIND_MARE_CALM_THRESHOLD:
            return "mare_1.png"
        elif wind_speed < WIND_MARE_MODERATE_THRESHOLD:
            return "mare_2.png"
        else:
            return "mare_3.png"
    else:
        giorno_anno = ora_locale.timetuple().tm_yday
        if SUMMER_START_DAY <= giorno_anno <= SUMMER_END_DAY:
            alba, tramonto = ALBA_SUMMER, TRAMONTO_SUMMER
        elif 20 <= giorno_anno < SUMMER_START_DAY:
            alba, tramonto = ALBA_WINTER_EARLY, TRAMONTO_WINTER_EARLY
        else:
            alba, tramonto = ALBA_WINTER_LATE, TRAMONTO_WINTER_LATE
            
        ora_decimale = ora_locale.hour + ora_locale.minute / 60
        is_day = alba <= ora_decimale <= tramonto
        suffisso = "_g" if is_day else "_n"

        if clct < CLOUD_CLEAR_THRESHOLD:
            cielo = "sereno"
        elif clct < CLOUD_PARTLY_THRESHOLD:
            cielo = "poconuv"
        elif clct < CLOUD_MOSTLY_THRESHOLD:
            cielo = "nuv"
        else:
            cielo = "cop"

        if tp < RAIN_LIGHT_THRESHOLD:
            return f"{cielo}{suffisso}.png"
        else:
            if t2m <= 0.5:
                if tp < RAIN_MEDIUM_THRESHOLD:
                    return f"{cielo}_1n{suffisso}.png"
                elif tp < RAIN_HEAVY_THRESHOLD:
                    return f"{cielo}_2n{suffisso}.png"
                else:
                    return f"{cielo}_3n{suffisso}.png"
            else:
                if tp < RAIN_MEDIUM_THRESHOLD:
                    return f"{cielo}_1p{suffisso}.png"
                elif tp < RAIN_HEAVY_THRESHOLD:
                    return f"{cielo}_2p{suffisso}.png"
                else:
                    return f"{cielo}_3p{suffisso}.png"

def get_last_sunday_of_month(year, month):
    """Calcola l'ultima domenica di un dato mese e anno."""
    d = datetime(year, month, 1)
    # Vai al primo giorno del mese successivo e poi indietro di un giorno per ottenere l'ultimo giorno del mese corrente
    # In questo modo si gestisce correttamente anche febbraio
    if month == 12:
        d = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = datetime(year, month + 1, 1) - timedelta(days=1)
    
    # Trova l'ultima domenica
    while d.weekday() != 6:  # 6 è Domenica
        d -= timedelta(days=1)
    return d

def get_local_time_offset(dt_utc_aware):
    """
    Determina l'offset dell'ora locale (solare +1, legale +2) per l'Italia
    basandosi sulla data fornita.
    L'ora legale inizia l'ultima domenica di Marzo, finisce l'ultima domenica di Ottobre.
    """
    year = dt_utc_aware.year

    # Calcola l'ultima domenica di Marzo
    dst_start = get_last_sunday_of_month(year, 3)
    # L'ora legale scatta alle 2 del mattino UTC (che diventano le 3 locali)
    dst_start = dst_start.replace(hour=2, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

    # Calcola l'ultima domenica di Ottobre
    dst_end = get_last_sunday_of_month(year, 10)
    # L'ora solare scatta alle 2 del mattino UTC (che diventano le 1 locali)
    dst_end = dst_end.replace(hour=2, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

    # Converti il datetime fornito in UTC se non lo è già , per confronto omogeneo
    if dt_utc_aware.tzinfo is None:
        dt_utc_aware = dt_utc_aware.replace(tzinfo=timezone.utc)
    else:
        dt_utc_aware = dt_utc_aware.astimezone(timezone.utc)

    if dst_start <= dt_utc_aware < dst_end:
        return 2 # Ora Legale
    else:
        return 1 # Ora Solare

def setup_pdf_document(pdf_obj, font_path):
    """
    Configura le proprietà  base del documento PDF, inclusi font e auto page break.
    """
    pdf_obj.set_auto_page_break(auto=True, margin=15)
    if os.path.exists(font_path):
        pdf_obj.add_font('DejaVu', '', font_path, uni=True)
        pdf_obj.set_font('DejaVu', '', NORMAL_FONT_SIZE)
    else:
        print(f"Attenzione: Font DejaVuSans.ttf non trovato a {font_path}. Verrà  utilizzato un font di default (Helvetica).")
        pdf_obj.set_font('Helvetica', '', NORMAL_FONT_SIZE)  

def add_page_header(pdf_obj, city_name, current_day, run_datetime_utc, logo_path=None):
    """
    Aggiunge l'intestazione della pagina, inclusi logo, titolo e data del run.
    """
    pdf_obj.add_page()
    
    if logo_path and os.path.exists(logo_path):
        try:
            with Image.open(logo_path) as img:
                width, height = img.size
                aspect_ratio = width / height
                logo_h = 15
                logo_w = logo_h * aspect_ratio
                pdf_obj.image(logo_path, x=10, y=10, w=logo_w, h=logo_h)
        except Exception as e:
            print(f"Attenzione: Errore caricamento logo da {logo_path}. Errore: {e}")

    days_of_week_it = {
        0: "Lunedì", 1: "Martedì", 2: "Mercoledì", 3: "Giovedì", 4: "Venerdì", 5: "Sabato", 6: "Domenica"
    }
    day_name = days_of_week_it[current_day.weekday()]

    pdf_obj.set_xy(10, 25)
    pdf_obj.set_font('DejaVu', '', HEADER_FONT_SIZE)
    pdf_obj.set_text_color(*COLOR_TEXT)
    pdf_obj.cell(0, 10, f"Bollettino Triorario - {city_name} - {day_name} {current_day.strftime('%d/%m/%Y')}", ln=True, align='C')
    
    pdf_obj.set_font('DejaVu', '', NORMAL_FONT_SIZE)
    run_date_display = run_datetime_utc.strftime('%d/%m/%Y %H')
    pdf_obj.cell(0, 7, f"ICON 2I: run {run_date_display}", ln=True, align='C')
    pdf_obj.ln(3)
    
    pdf_obj.set_font('DejaVu', '', NORMAL_FONT_SIZE)

def add_table_header(pdf_obj, headers, col_widths):
    """
    Aggiunge l'intestazione della tabella con colori e bordi.
    """
    pdf_obj.set_fill_color(*COLOR_HEADER_BG)
    pdf_obj.set_text_color(*COLOR_TEXT)
    pdf_obj.set_draw_color(*COLOR_BORDER)
    pdf_obj.set_line_width(BORDER_LINE_WIDTH)
    
    table_width = sum(col_widths)
    page_width = pdf_obj.w
    start_x = (page_width - table_width) / 2
    pdf_obj.set_x(start_x)

    for header, width in zip(headers, col_widths):
        pdf_obj.cell(width, 7, header, border=1, align='C', fill=True)
    pdf_obj.ln()
    return start_x

def add_table_row(pdf_obj, row_data, col_widths, start_x, row_index, icons_path):
    """
    Aggiunge una singola riga di dati alla tabella, inclusa l'icona meteo.
    """
    if row_index % 2 == 0:
        pdf_obj.set_fill_color(*COLOR_ROW_EVEN_BG)
    else:
        pdf_obj.set_fill_color(*COLOR_ROW_ODD_BG)

    pdf_obj.set_text_color(*COLOR_TEXT)
    pdf_obj.set_font('DejaVu', '', NORMAL_FONT_SIZE)

    pdf_obj.set_x(start_x)    
    current_y = pdf_obj.get_y()

    # Applica l'ora locale qui per la visualizzazione nella cella
    # Nota: row_data['Ora'] contiene già  l'ora locale grazie alle modifiche in generate_weather_bulletin
    pdf_obj.cell(col_widths[0], CELL_HEIGHT, row_data['Ora'], border=1, align='C', fill=True)
    
    icon_cell_x = pdf_obj.get_x()
    pdf_obj.cell(col_widths[1], CELL_HEIGHT, "", border=1, fill=True)

    values = [
        row_data["Temperatura (°C)"],
        row_data["Umidità (%)"],
        row_data["Precipitazione (mm)"],
        row_data["Nuvolosità (%)"],
        row_data["Pressione (hPa)"],
        row_data["Vento (km/h)"],
        row_data["Direzione Vento"]
    ]
    for val, width in zip(values, col_widths[2:]):
        pdf_obj.cell(width, CELL_HEIGHT, str(val), border=1, align='C', fill=True)
    
    pdf_obj.ln()

    icon_full_path = get_cached_icon_path(row_data["Icona"])
    if icon_full_path:
        try:
            img_center_x = icon_cell_x + (col_widths[1] - ICON_DISPLAY_WIDTH) / 2
            img_center_y = current_y + (CELL_HEIGHT - ICON_DISPLAY_HEIGHT) / 2
            pdf_obj.image(icon_full_path, x=img_center_x, y=img_center_y, w=ICON_DISPLAY_WIDTH, h=ICON_DISPLAY_HEIGHT)
        except RuntimeError as e:
            print(f"Attenzione: Errore durante l'inserimento dell'immagine {icon_full_path}. Errore: {e}")
            pdf_obj.set_xy(icon_cell_x, current_y)
            pdf_obj.set_font_size(SMALL_FONT_SIZE)
            pdf_obj.cell(col_widths[1], CELL_HEIGHT, "Icona N/D", border=0, align='C', fill=False)
            pdf_obj.set_font_size(NORMAL_FONT_SIZE)

def add_daily_summary(pdf_obj, daily_summary_data, icons_path):
    """
    Aggiunge una sezione di riepilogo giornaliero (Icona, Min Temp, Max Temp, Precipitazione Totale) centrata.
    """
    pdf_obj.ln(5) # Spazio prima del riepilogo
    
    # Calcola la posizione X per centrare il blocco del riepilogo
    fixed_summary_block_width = 120 # Larghezza totale desiderata per il blocco del riepilogo (es. 120mm)
    summary_block_start_x = (pdf_obj.w - fixed_summary_block_width) / 2
    pdf_obj.set_x(summary_block_start_x)

    pdf_obj.set_font('DejaVu', '', NORMAL_FONT_SIZE)
    pdf_obj.set_fill_color(*COLOR_ROW_EVEN_BG) # Sfondo per il riepilogo
    
    summary_cell_height = CELL_HEIGHT * 1.5

    # Calcola l'icona del giorno
    icon_filename = get_weather_icon_filename(
        daily_summary_data['Nuvolosità  Media'],
        daily_summary_data['Precipitazione Totale'],
        daily_summary_data['Temperatura Media'],
        daily_summary_data['Ora Riferimento'],
        daily_summary_data['Vento Medio'],
        daily_summary_data['City Name']
    )
    icon_full_path = get_cached_icon_path(icon_filename)

    # Definiamo le larghezze delle singole celle all'interno del riepilogo
    icon_summary_cell_width = 30
    temp_min_summary_cell_width = 30
    temp_max_summary_cell_width = 30
    precip_summary_cell_width = 30

    # Disegna le celle del riepilogo
    current_x = summary_block_start_x
    current_y_for_text = pdf_obj.get_y() # Salva la Y per allineare il testo verticalmente al centro

    # Cella Icona
    pdf_obj.set_xy(current_x, current_y_for_text)
    pdf_obj.cell(icon_summary_cell_width, summary_cell_height, "", border=1, fill=True, align='C')
    if icon_full_path:
        img_center_x = current_x + (icon_summary_cell_width - ICON_DISPLAY_WIDTH) / 2
        img_center_y = current_y_for_text + (summary_cell_height - ICON_DISPLAY_HEIGHT) / 2
        try:
            pdf_obj.image(icon_full_path, x=img_center_x, y=img_center_y, w=ICON_DISPLAY_WIDTH, h=ICON_DISPLAY_HEIGHT)
        except RuntimeError as e:
            print(f"Attenzione: Errore durante l'inserimento dell'immagine {icon_full_path} nel riepilogo. Errore: {e}")
            pdf_obj.set_xy(img_center_x, current_y_for_text)
            pdf_obj.set_font_size(SMALL_FONT_SIZE)
            pdf_obj.cell(icon_summary_cell_width, summary_cell_height, "Icona N/D", border=0, align='C', fill=False)
            pdf_obj.set_font_size(NORMAL_FONT_SIZE)
    current_x += icon_summary_cell_width

    # Cella Temperatura Minima
    pdf_obj.set_xy(current_x, current_y_for_text)
    pdf_obj.cell(temp_min_summary_cell_width, summary_cell_height, "", border=1, fill=True, align='C')
    pdf_obj.set_xy(current_x, current_y_for_text + (summary_cell_height - pdf_obj.font_size) / 2)
    pdf_obj.set_text_color(*COLOR_TEMP_MIN)
    pdf_obj.cell(temp_min_summary_cell_width, pdf_obj.font_size, f"Min: {daily_summary_data['Temperatura Minima']:.0f}°C", 0, 0, 'C')
    current_x += temp_min_summary_cell_width

    # Cella Temperatura Massima
    pdf_obj.set_xy(current_x, current_y_for_text)
    pdf_obj.cell(temp_max_summary_cell_width, summary_cell_height, "", border=1, fill=True, align='C')
    pdf_obj.set_xy(current_x, current_y_for_text + (summary_cell_height - pdf_obj.font_size) / 2)
    pdf_obj.set_text_color(*COLOR_TEMP_MAX)
    pdf_obj.cell(temp_max_summary_cell_width, pdf_obj.font_size, f"Max: {daily_summary_data['Temperatura Massima']:.0f}°C", 0, 0, 'C')
    current_x += temp_max_summary_cell_width

    # Cella Precipitazione Totale
    pdf_obj.set_xy(current_x, current_y_for_text)
    pdf_obj.cell(precip_summary_cell_width, summary_cell_height, "", border=1, fill=True, align='C')
    pdf_obj.set_xy(current_x, current_y_for_text + (summary_cell_height - pdf_obj.font_size) / 2)
    pdf_obj.set_text_color(*COLOR_PRECIP)
    pdf_obj.cell(precip_summary_cell_width, pdf_obj.font_size, f"Prec: {daily_summary_data['Precipitazione Totale']:.1f} mm", 0, 0, 'C')
    current_x += precip_summary_cell_width

    pdf_obj.ln(summary_cell_height + 5) # Spazio dopo il riepilogo
    pdf_obj.set_font_size(NORMAL_FONT_SIZE) # Reset font size
    pdf_obj.set_text_color(*COLOR_TEXT) # Reset colore testo


# Funzione Principale di Generazione Bollettino
def generate_weather_bulletin(city_name, capoluoghi_dati, run_datetime_utc, output_dir, icons_path, font_path, logo_path=None):
    """
    Genera un bollettino meteorologico triorario in formato PDF per una data città .
    """
    dati_citta = capoluoghi_dati.get(city_name)
    if not dati_citta:
        print(f"Errore: Dati non trovati per la città  '{city_name}'.")
        return

    ore_previsione = pd.date_range(run_datetime_utc, periods=NUM_HOURS_FORECAST, freq='h')

    truncated_dati = {
        k: v[:NUM_HOURS_FORECAST] if isinstance(v, (list, np.ndarray)) else v
        for k, v in dati_citta.items()
    }

    df = pd.DataFrame({
        'Data/Ora': ore_previsione,
        'Temperatura (°C)': truncated_dati['t2m'],
        'Umidità (%)': truncated_dati['rh2m'],
        'Precipitazione (mm)': truncated_dati['tp'],
        'Nuvolosità (%)': truncated_dati['clct'],
        'Pressione (hPa)': truncated_dati['pmsl'],
        'Vento (km/h)': truncated_dati['wind_speed'],
        'Direzione Vento': truncated_dati['wind_dir_cardinal'],
    })

    # Calcolo riassunto giornaliero dal dataframe orario completo
    daily_summaries = {}
    for day, day_group in df.groupby(df['Data/Ora'].dt.date):
        # Per l'ora di riferimento del riepilogo giornaliero, usiamo l'ora locale del 12-esimo valore
        # o l'ultima disponibile se il gruppo è più piccolo.
        ref_time_utc = day_group['Data/Ora'].iloc[min(12, len(day_group) -1)]
        offset = get_local_time_offset(ref_time_utc)
        ref_time_local = ref_time_utc + timedelta(hours=offset)

        daily_summaries[day] = {
            'Temperatura Minima': day_group['Temperatura (°C)'].min(),
            'Temperatura Massima': day_group['Temperatura (°C)'].max(),
            'Nuvolosità  Media': day_group['Nuvolosità (%)'].mean(),
            'Precipitazione Totale': day_group['Precipitazione (mm)'].sum(),
            'Temperatura Media': day_group['Temperatura (°C)'].mean(),
            'Vento Medio': day_group['Vento (km/h)'].mean(),
            'Ora Riferimento': ref_time_local.to_pydatetime(), # Passiamo l'ora locale per l'icona
            'City Name': city_name
        }

    # Raggruppamento Triorario
    df['Grouping_Key'] = df['Data/Ora'].dt.floor(f'{GROUPING_INTERVAL_HOURS}h')

    grouped_data_list = []
    for key, group in df.groupby('Grouping_Key'):
        ora_utc_triora = group['Data/Ora'].iloc[0]
        
        # Calcola l'offset per questa specifica ora trioraria
        offset = get_local_time_offset(ora_utc_triora.to_pydatetime())
        ora_locale_triora = ora_utc_triora + timedelta(hours=offset) # Ora da mostrare e usare per icona giorno/notte
            
        avg_temp = group['Temperatura (°C)'].mean()
        avg_rh = group['Umidità (%)'].mean()
        sum_tp = group['Precipitazione (mm)'].sum()
        avg_clct = group['Nuvolosità (%)'].mean()
        avg_pmsl = group['Pressione (hPa)'].mean()
        avg_wind_speed = group['Vento (km/h)'].mean()

        wind_dir_cardinal = group['Direzione Vento'].iloc[0]
        icon_clct = avg_clct
        icon_tp = sum_tp
        icon_t2m = avg_temp
        icon_wind_speed = avg_wind_speed

        weather_icon_filename = get_weather_icon_filename(
            icon_clct, icon_tp, icon_t2m, ora_locale_triora.to_pydatetime(), icon_wind_speed, city_name
        )

        grouped_data_list.append({
            'Data/Ora': ora_utc_triora, # Manteniamo l'originale per il raggruppamento del giorno
            'Ora': ora_locale_triora.strftime("%H:%M"), # Questa è l'ora locale da visualizzare
            'Icona': weather_icon_filename,
            'Temperatura (°C)': avg_temp,
            'Umidità (%)': avg_rh,
            'Precipitazione (mm)': sum_tp,
            'Nuvolosità (%)': avg_clct,
            'Pressione (hPa)': avg_pmsl,
            'Vento (km/h)': avg_wind_speed,
            'Direzione Vento': wind_dir_cardinal
        })

    df_triorario = pd.DataFrame(grouped_data_list)

    # Formattazione dei valori numerici per la visualizzazione nel PDF
    df_triorario["Temperatura (°C)"] = df_triorario["Temperatura (°C)"].round(1).astype(str)
    df_triorario["Umidità (%)"] = df_triorario["Umidità (%)"].round(0).astype(int).astype(str)
    df_triorario["Precipitazione (mm)"] = df_triorario["Precipitazione (mm)"].round(1).astype(str)
    df_triorario["Nuvolosità (%)"] = df_triorario["Nuvolosità (%)"].round(0).astype(int).astype(str)
    df_triorario["Pressione (hPa)"] = df_triorario["Pressione (hPa)"].round(0).astype(int).astype(str)
    df_triorario["Vento (km/h)"] = df_triorario["Vento (km/h)"].round(1).astype(str)

    # Raggruppa per giorno (del bollettino triorario)
    df_triorario['Giorno'] = df_triorario['Data/Ora'].dt.date
    giorni_raggruppati_triorario = dict(tuple(df_triorario.groupby('Giorno')))

    # Inizializza il documento PDF
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    setup_pdf_document(pdf, font_path)

    # Modifica l'intestazione della colonna
    headers = ["Ora Locale", "Icona", "T (°C)", "UR (%)", "Prec. (mm)", "Nuv. (%)", "Press. (hPa)", "Vento (km/h)", "Dir"]
    col_widths = [20, 25, 25, 25, 25, 25, 30, 25, 20]

    for i, (giorno, dati_giorno) in enumerate(giorni_raggruppati_triorario.items()):
        add_page_header(pdf, city_name, giorno, run_datetime_utc, logo_path)
        
        # Aggiungi il riepilogo giornaliero qui, prima dell'intestazione della tabella e delle righe
        if giorno in daily_summaries:
            add_daily_summary(pdf, daily_summaries[giorno], icons_path)
            # Aggiungi spazio dopo il riepilogo e prima della tabella oraria
            pdf.ln(5)

        start_x = add_table_header(pdf, headers, col_widths)

        # Aggiungi le righe di dati triorari
        for row_index, (_, row) in enumerate(dati_giorno.iterrows()):
            add_table_row(pdf, row, col_widths, start_x, row_index, icons_path)

    # Generazione nome file con data e ora del run
    run_date_str = run_datetime_utc.strftime('%Y%m%d%H')
    output_filename = os.path.join(output_dir, f"{city_name.lower()}_{run_date_str}.pdf")
    # output_filename = os.path.join(output_dir, f"bollettino_triorario_{city_name.lower()}_{run_date_str}_corr.pdf")
    
    pdf.output(output_filename)
    print(f"Bollettino triorario generato con successo: {output_filename}")

# Esecuzione Principale
if __name__ == "__main__":
    base_dir = os.getcwd()  # directory di lavoro corrente nel runner GitHub Actions
    output_dir = os.path.join(base_dir, "output")
    data_dir = os.path.join(base_dir, "data")
    pickle_data_path = os.path.join(data_dir, 'capoluoghi_dati.pkl')
    try:
        with open(pickle_data_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        capoluoghi_data = loaded_data['capoluoghi_dati']
        run_datetime_utc = loaded_data['run_datetime_utc']

        run_folder = run_datetime_utc.strftime('%Y%m%d')
        OUTPUT_DIR = os.path.join(output_dir, run_folder)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        cities_to_process = [
            "Montesilvano",
            "Ghiacciaio Calderone",
            # "Ristoro Mucciante",
            "Campo Imperatore",
            "Caramanico Terme (PE) - 650 m",
            # "Bivacco Alpe Mottac - 1690 m",
            # "Castelnuovo Magra (SP) - 190 m",
            
            # "Scanno (AQ) - 1050 m",
            # "Malta",
            # "Sassari",
            # "Casteldelci (RN) - 632 m",
            # "Bolsena (VT) - 350 m",
            # "Tuscania (VT) - 165 m",
            "Mirabella Eclano",
            "Tropea",
            
            "Bologna",
            "Aosta",
            "Torino",
            "Milano",
            "Trento",
            "Bolzano",
            "Venezia",
            "Trieste",
            "Genova",
            "Firenze",
            "Perugia",
            "Ancona",
            "L'Aquila",
            "Campobasso",
            "Napoli",
            "Bari",
            "Potenza",
            "Catanzaro",
            "Palermo",
            "Cagliari",
            "Roma",
            "Faenza",
            "Verona",
            "Padova",
            "Pescasseroli",
            "Nulvi",
            "Sutera",
            "Montecosaro"
        ]

        for city in cities_to_process:
            print(f"Generazione bollettino per {city}...")
            generate_weather_bulletin(city, capoluoghi_data, run_datetime_utc, OUTPUT_DIR, ICONS_PATH, FONT_PATH, LOGO_PATH)

    except FileNotFoundError:
        print(f"Errore: Il file dati '{pickle_data_path}' non è stato trovato. Assicurati che i dati siano stati generati e salvati in precedenza.")
    except Exception as e:
        print(f"Si è verificato un errore inatteso durante l'esecuzione: {e}")
