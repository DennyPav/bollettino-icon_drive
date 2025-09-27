#!/usr/bin/env python3
"""
Bollettino giornaliero con sistema di First Guess da ICON 2I
Versione completa con percorsi relativi al repository e generazione automatica di grafici e PDF.
"""

import os
from pathlib import Path
import requests
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from shapely.geometry import box
from shapely.ops import unary_union
import pandas as pd
from fpdf import FPDF
from PIL import Image

# === CONFIG (percorsi relativi alla root del repository) ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
ICONS_DIR = BASE_DIR / "icons"
FONTS_DIR = BASE_DIR / "fonts"

SATELLITE_PATH = ICONS_DIR / "satellite.tif"
LOGO_PATH = ICONS_DIR / "logo.png"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ICONS_DIR.mkdir(parents=True, exist_ok=True)
FONTS_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_FONTS = [
    FONTS_DIR / "DejaVuSans.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
]
FONT_PATH = next((p for p in CANDIDATE_FONTS if p.exists()), None)

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
    "Castelnuovo Magra (SP) - 190 m": (44.099743, 10.017324, 190)
}

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
    for var in VARIABLES:
        var_url = f'{base_url}{var}/'
        try:
            r = requests.get(var_url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            grib_files = [a.get('href') for a in soup.find_all('a') if a.get('href', '').endswith('.grib')]
            if not grib_files:
                continue
            file_url = var_url + grib_files[0]
            local_path = DATA_DIR / f'{var}.grib'

            skip = False
            if local_path.exists():
                head = requests.head(file_url, timeout=30)
                if 'Last-Modified' in head.headers:
                    remote = datetime.strptime(head.headers['Last-Modified'], '%a, %d %b %Y %H:%M:%S %Z')
                    local = datetime.utcfromtimestamp(local_path.stat().st_mtime)
                    if local >= remote:
                        print(f'{var} già aggiornato.')
                        skip = True

            if not skip:
                with requests.get(file_url, stream=True, timeout=120) as resp:
                    with open(local_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f'Scaricato {var}')
        except Exception as e:
            print(f'Errore download {var}: {e}')

def load_data():
    data = {}
    for var in VARIABLES:
        path = DATA_DIR / f'{var}.grib'
        if path.exists():
            try:
                ds = xr.open_dataset(str(path), engine='cfgrib')
                data[var] = ds
            except Exception as e:
                print(f'Errore lettura {var}: {e}')
    return data

def extract_variable(var, lat_idx, lon_idx):
    if var.ndim == 3:
        center = var[:, lat_idx, lon_idx]
        first_ring = [var[:, lat_idx + i, lon_idx + j]
                      for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)]
        second_ring = [var[:, lat_idx + i, lon_idx + j]
                       for i in [-2, -1, 0, 1, 2]
                       for j in [-2, -1, 0, 1, 2]
                       if (abs(i) == 2 or abs(j) == 2)]
        first_ring = np.stack(first_ring, axis=0).mean(axis=0)
        second_ring = np.stack(second_ring, axis=0).mean(axis=0) if second_ring else 0
        return 0.5 * center + 0.25 * first_ring + 0.25 * second_ring
    elif var.ndim == 2:
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
        lapse_rate = 0.0065
        p_lapse_rate = 12

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

def save_capoluoghi_pickle(capoluoghi_dati, run_datetime_utc):
    pickle_path = OUTPUT_DIR / 'capoluoghi_dati.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump({'capoluoghi_dati': capoluoghi_dati, 'run_datetime_utc': run_datetime_utc}, f)
    print(f'Dati salvati in {pickle_path}')

def plot_and_generate_pdf(capoluoghi_dati, run_datetime_utc):
    fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([6, 19, 36, 47], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.3)
    ax.add_feature(cfeature.RIVERS, alpha=0.3)

    for city, d in capoluoghi_dati.items():
        ax.plot(d['lon'], d['lat'], marker='o', color='red', markersize=2, transform=ccrs.PlateCarree())
        ax.text(d['lon']+0.1, d['lat']+0.1, f"{city}\n{d['t2m'][0]}°C", fontsize=5, transform=ccrs.PlateCarree())

    plt.title(f"First Guess ICON2I - Run {run_datetime_utc.strftime('%Y-%m-%d %H UTC')}")
    img_path = OUTPUT_DIR / f"mappa_{run_datetime_utc.strftime('%Y%m%d%H')}.png"
    plt.savefig(img_path, dpi=200)
    plt.close()

    pdf = FPDF()
    pdf.add_page()
    if LOGO_PATH.exists():
        pdf.image(str(LOGO_PATH), x=10, y=8, w=33)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f"Bollettino ICON2I - {run_datetime_utc.strftime('%d/%m/%Y %H UTC')}", ln=1)
    pdf.image(str(img_path), x=10, y=30, w=180)
    pdf.output(str(OUTPUT_DIR / f"bollettino_{run_datetime_utc.strftime('%Y%m%d%H')}.pdf"))

def main():
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
        save_capoluoghi_pickle(capoluoghi_dati, run_datetime_utc)
        plot_and_generate_pdf(capoluoghi_dati, run_datetime_utc)
    else:
        print("Nessun dato disponibile.")

if __name__ == "__main__":
    main()
