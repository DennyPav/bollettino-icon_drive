"""
Bollettino giornaliero con sistema di First Guess da ICON 2I, aggiornamento giornaliero per i RUN 00 e 12

@author: deniel
"""

import os
import requests
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo
import pickle
import shutil
import math
from collections import Counter
import pytz
import locale
import rasterio
import shapely.geometry as sgeom
import cartopy.io.shapereader as shpreader
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.colors import to_rgba
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.image import imread
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import boto3
import io
from PIL import Image

# === CONFIG ===
DATA_DIR = os.path.join(os.getcwd(), "data")
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
RUN_HOURS = ['00', '12']
VARIABLES = ['T_2M', 'RELHUM', 'TOT_PREC', 'CLCT', 'PMSL', 'U_10M', 'V_10M', 'HSURF', 'CLCL', 'CLCM', 'CLCH', 'LPI', 'CAPE_ML', 'CAPE_CON', 'UH_MAX']
CAPOLUOGHI = {
    'Ancona': (43.6158, 13.5189, 16),
    'Aosta': (45.7370, 7.3201, 583),
    'Sassari': (40.7259, 8.5555, 225),
    'Bari': (41.1171, 16.8719, 5),
    'Catania': (37.5079, 15.0830, 7),
    'Lecce': (40.3529, 18.1743, 49),
    'Livigno': (46.5382, 10.1413, 1816), 
    'Olbia': (40.9235, 9.4989, 15),
    'Elba': (42.8142, 10.3160, 344),
    'Foggia': (41.4622, 15.5446, 76),
    'Livorno': (43.5485, 10.3106, 3),
    'Messina': (38.1938, 15.5540, 3),
    'Mar_Tirreno': (39.753155, 12.0000, 0),
    'Mar_Adriatico': (42.974424, 15.278642, 0),
    'Mar_Ionio': (37.965580, 17.944965, 0),
    'Formazza': (46.372819, 8.426935, 1380),
    'Pantelleria': (36.8335, 11.9474, 836),
    'Parma': (44.8015, 10.3279, 57),
    'Cortina d Ampezzo': (46.5405, 12.1357, 1224),
    'Pescara': (42.4643, 14.2134, 4),
    'Rimini': (44.0604, 12.5653, 6),
    'Bologna': (44.4949, 11.3426, 54),
    'Bolzano': (46.4983, 11.3548, 262),
    'Cagliari': (39.2238, 9.1217, 6),
    'Campobasso': (41.5600, 14.6600, 701),
    'Catanzaro': (38.8896, 16.6052, 320),
    'Firenze': (43.7696, 11.2558, 50),
    'Genova': (44.4056, 8.9463, 19),
    'L Aquila': (42.3498, 13.3995, 714),
    'Milano': (45.4642, 9.1900, 120),
    'Napoli': (40.8518, 14.2681, 17),
    'Palermo': (38.1157, 13.3615, 14),
    'Perugia': (43.1122, 12.3888, 493),
    'Potenza': (40.6401, 15.8050, 819),
    'Reggio Calabria': (38.1105, 15.6613, 31),
    'Roma': (41.9028, 12.4964, 21),
    'Torino': (45.0703, 7.6869, 239),
    'Trento': (46.0700, 11.1200, 194),
    'Trieste': (45.6495, 13.7768, 2),
    'Venezia': (45.4408, 12.3155, 2)
}

SEASON_THRESHOLDS = {
    "winter": {"start_day": 1, "end_day": 80, "fog_rh": 96, "haze_rh": 85, "fog_wind": 7.0, "haze_wind": 12.0, "fog_max_t": 15.0},
    "spring": {"start_day": 81, "end_day": 172, "fog_rh": 97, "haze_rh": 85, "fog_wind": 6.0, "haze_wind": 10.0, "fog_max_t": 20.0},
    "summer": {"start_day": 173, "end_day": 263, "fog_rh": 98, "haze_rh": 90, "fog_wind": 4.0, "haze_wind": 9.0, "fog_max_t": 26.0},
    "autumn": {"start_day": 264, "end_day": 365, "fog_rh": 95, "haze_rh": 88, "fog_wind": 7.0, "haze_wind": 11.0, "fog_max_t": 20.0}
}

# === UTILS ===

def kelvin_to_celsius(k): return k - 273.15
def wind_speed_direction(u, v): return np.sqrt(u**2 + v**2) * 3.6, (np.degrees(np.arctan2(-u, -v)) % 360)
def wind_dir_to_cardinal(deg):
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    return dirs[int((deg + 22.5) % 360 // 45)]

def get_season_precise(dt):
    d = dt.timetuple().tm_yday
    for s, t in SEASON_THRESHOLDS.items():
        if t["start_day"] <= d <= t["end_day"]: return s, t
    return "winter", SEASON_THRESHOLDS["winter"]

def wet_bulb_celsius(t_c, rh_percent):
    return t_c * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659)) + np.arctan(t_c + rh_percent) - np.arctan(rh_percent - 1.676331) + 0.00391838 * rh_percent**1.5 * np.arctan(0.023101 * rh_percent) - 4.686035

def download_icon_data():
    now = datetime.utcnow()
    if now.hour < 2:
        run_hour = '12'; run_date = (now - timedelta(days=1)).strftime('%Y%m%d')
    elif now.hour < 14:
        run_hour = '00'; run_date = now.strftime('%Y%m%d')
    else:
        run_hour = '12'; run_date = now.strftime('%Y%m%d')

    os.makedirs(DATA_DIR, exist_ok=True)

    for var in VARIABLES:
        if var.lower() == 'hsurf':
            local_path = os.path.join(os.path.dirname(__file__), 'icon_2I_h_surface.grib')
            dest_path = os.path.join(DATA_DIR, 'HSURF.grib')
            if not os.path.exists(local_path):
                print(f"ERRORE: {local_path} non trovato!")
                continue
            shutil.copy(local_path, dest_path)
            print(f"HSURF: copiato {local_path} → {dest_path}")
            continue
            
        base_url = f'https://meteohub.agenziaitaliameteo.it/nwp/ICON-2I_SURFACE_PRESSURE_LEVELS/{run_date}{run_hour}/{var}/'
        try:
            r = requests.get(base_url)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            grib_files = [a.get('href') for a in soup.find_all('a') if a.get('href', '').endswith('.grib')]

            if not grib_files:
                print(f"ATTENZIONE: nessun file grib trovato per {var}")
                continue

            file_url = base_url + grib_files[0]
            local_path = os.path.join(DATA_DIR, f'{var}.grib')

            skip = False
            if os.path.exists(local_path):
                head = requests.head(file_url)
                if 'Last-Modified' in head.headers:
                    remote = datetime.strptime(head.headers['Last-Modified'], '%a, %d %b %Y %H:%M:%S %Z')
                    local = datetime.utcfromtimestamp(os.path.getmtime(local_path))
                    if local >= remote:
                        print(f'{var} già aggiornato.'); skip = True
            
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

# INTERPOLAZIONE SPAZIALE ESATTA COME IN ICON2I
def extract_variable(var, y, x, weighted=False):
    if var.ndim == 3: # (time, lat, lon)
        if not weighted:
            return var[:, y, x]
        NY, NX = var.shape[1], var.shape[2]
        slices = []
        weights = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if 0 <= y+di < NY and 0 <= x+dj < NX:
                    w = 1.0 if (di==0 and dj==0) else (0.5 if di==0 or dj==0 else 0.25)
                    weights.append(w)
                    slices.append(var[:, y+di, x+dj])
        return np.average(np.stack(slices, axis=0), axis=0, weights=weights)
        
    elif var.ndim == 2: # (lat, lon)
        if not weighted:
            return var[y, x]
        NY, NX = var.shape[0], var.shape[1]
        val, wtot = 0, 0
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if 0 <= y+di < NY and 0 <= x+dj < NX:
                    w = 1.0 if (di==0 and dj==0) else (0.5 if di==0 or dj==0 else 0.25)
                    val += var[y+di, x+dj] * w
                    wtot += w
        return val / wtot
    else:
        raise ValueError(f"Array shape non gestita: {var.shape}")

# LOGICA METEO ICON2I
def classify_weather_hourly(t2m, rh2m, clct, clcl, clcm, clch,
                            tp_rate, wind_kmh, lpi, cape, uh,
                            season, season_thresh):

    octas = clct / 100.0 * 8
    low = clcl if np.isfinite(clcl) else (clcm if np.isfinite(clcm) else 0)

    if clch > 60 and low < 30 and octas > 5: c_state = "NUBI ALTE"
    elif octas <= 2: c_state = "SERENO"
    elif octas <= 4: c_state = "POCO NUVOLOSO"
    elif octas <= 6: c_state = "NUVOLOSO"
    else: c_state = "COPERTO"

    wet_bulb = wet_bulb_celsius(t2m, rh2m)
    is_snow = wet_bulb < 0.5
    prec_high = "NEVE" if is_snow else "PIOGGIA"
    prec_low = "NEVISCHIO" if is_snow else "PIOGGERELLA"

    conv_signal = ((cape >= 400 and lpi >= 1.5) or (uh >= 50) or (cape >= 800))
    rain_signal = tp_rate >= 0.3
    gust_signal = wind_kmh >= 35
    deep_clouds = clct >= 90 and (clcm >= 40 or clch >= 40)

    if conv_signal and (rain_signal or gust_signal) and deep_clouds:
        if c_state == "SERENO": c_state = "POCO NUVOLOSO"
        return f"{c_state} TEMPORALE"

    tp_rate = round(tp_rate, 1)

    fog_rh = season_thresh.get("fog_rh", 95)
    fog_wd = season_thresh.get("fog_wind", 8)
    fog_t = season_thresh.get("fog_max_t", 18)
    haze_rh = season_thresh.get("haze_rh", 85)
    haze_wd = season_thresh.get("haze_wind", 12)

    if tp_rate >= 0.1:
        if c_state == "SERENO": c_state = "POCO NUVOLOSO"
        if tp_rate > 0.3:
            intent = "INTENSA" if tp_rate >= 7.0 else ("MODERATA" if tp_rate >= 2.0 else "DEBOLE")
            return f"{c_state} {prec_high} {intent}"
        elif math.isclose(tp_rate, 0.3, abs_tol=1e-3):
            if c_state == "NUBI ALTE": c_state = "COPERTO"
            return f"{c_state} {prec_low}"
        else:
            if t2m < fog_t and rh2m >= fog_rh and wind_kmh <= fog_wd and low >= 80: return "NEBBIA"
            elif t2m < fog_t and rh2m >= haze_rh and wind_kmh <= haze_wd and low >= 50: return "FOSCHIA"
            else:
                if c_state == "NUBI ALTE": c_state = "COPERTO"
                return f"{c_state} {prec_low}"
    else:
        if t2m < fog_t and rh2m >= fog_rh and wind_kmh <= fog_wd and low >= 80: return "NEBBIA"
        elif t2m < fog_t and rh2m >= haze_rh and wind_kmh <= haze_wd and low >= 50: return "FOSCHIA"
        else: return c_state

def classify_daily_weather(recs, clct_avg, clcl_avg, clcm_avg, clch_avg, tp_tot, season, thresh):
    snow_hours = 0
    rain_hours = 0
    has_significant_snow_or_rain = False
    fog_hours = 0
    haze_hours = 0
    has_storm = False

    for r in recs:
        hour = int(r.get("h", 0))
        wtxt = r.get("w", "")

        if "TEMPORALE" in wtxt: has_storm = True

        if "PIOGGIA" in wtxt:
            has_significant_snow_or_rain = True
            rain_hours += 1
        elif "NEVE" in wtxt:
            has_significant_snow_or_rain = True
            snow_hours += 1
        elif 5 <= hour <= 22:
            if "NEBBIA" in wtxt: fog_hours += 1
            elif "FOSCHIA" in wtxt: haze_hours += 1

    is_snow_day = snow_hours > rain_hours
    octas = clct_avg / 100.0 * 8
    low = clcl_avg if np.isfinite(clcl_avg) else (clcm_avg if np.isfinite(clcm_avg) else 0)

    if clch_avg > 60 and low < 30 and octas > 5: c_state = "NUBI ALTE"
    elif octas <= 2: c_state = "SERENO"
    elif octas <= 4: c_state = "POCO NUVOLOSO"
    elif octas <= 6: c_state = "NUVOLOSO"
    else: c_state = "COPERTO"

    if has_storm:
        if c_state == "SERENO": c_state = "POCO NUVOLOSO"
        return f"{c_state} TEMPORALE"

    if not has_significant_snow_or_rain:
        total_fog_haze = fog_hours + haze_hours
        if total_fog_haze >= 9:
            return "NEBBIA" if fog_hours >= haze_hours else "FOSCHIA"
        else:
            return c_state

    prec_type = "NEVE" if is_snow_day else "PIOGGIA"
    if tp_tot >= 30.0: intensity = "INTENSA"
    elif tp_tot >= 10.0: intensity = "MODERATA"
    else: intensity = "DEBOLE"

    if c_state == "SERENO": c_state = "POCO NUVOLOSO"
    return f"{c_state} {prec_type} {intensity}"

def weather_data(data):
    t2m_raw = kelvin_to_celsius(data['T_2M']['t2m'].values)
    rh2m_raw = data['RELHUM']['r'].values
    tp_raw = np.diff(data['TOT_PREC']['tp'].values, axis=0, prepend=0)
    clct_raw = data['CLCT']['clct'].values
    pmsl_raw = data['PMSL']['pmsl'].values / 100
    u10_raw = data['U_10M']['u10'].values
    v10_raw = data['V_10M']['v10'].values
    hsurf_raw = data['HSURF']['hsurf'].values

    clcl_raw = data['CLCL']['ccl'].values if 'CLCL' in data else np.zeros_like(clct_raw)
    clcm_raw = data['CLCM']['ccl'].values if 'CLCM' in data else np.zeros_like(clct_raw)
    clch_raw = data['CLCH']['ccl'].values if 'CLCH' in data else np.zeros_like(clct_raw)
    lpi_raw = data['LPI']['unknown'].values if 'LPI' in data else np.zeros_like(clct_raw)
    cape_ml_raw = data['CAPE_ML']['cape_ml'].values if 'CAPE_ML' in data else np.zeros_like(clct_raw)
    cape_con_raw = data['CAPE_CON']['cape_con'].values if 'CAPE_CON' in data else np.zeros_like(clct_raw)
    cape_raw = np.maximum(cape_ml_raw, cape_con_raw)
    uh_raw = data['UH_MAX']['unknown'].values if 'UH_MAX' in data else np.zeros_like(clct_raw)

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

        hs_loc = extract_variable(hsurf_raw, lat_idx, lon_idx, False)
        delta_z = -alt_c + hs_loc

        lapse_rate = 0.0065
        p_lapse_rate = 12

        # Estrazione con Pesi dinamici
        t2m = extract_variable(t2m_raw, lat_idx, lon_idx, False) + lapse_rate * delta_z
        rh2m = np.clip(extract_variable(rh2m_raw, lat_idx, lon_idx, False), 0, 100)
        tp = extract_variable(tp_raw, lat_idx, lon_idx, True)
        clct = extract_variable(clct_raw, lat_idx, lon_idx, True)
        pmsl = extract_variable(pmsl_raw, lat_idx, lon_idx, False) + (delta_z / 100) * p_lapse_rate
        
        clcl = extract_variable(clcl_raw, lat_idx, lon_idx, True)
        clcm = extract_variable(clcm_raw, lat_idx, lon_idx, True)
        clch = extract_variable(clch_raw, lat_idx, lon_idx, True)
        lpi = extract_variable(lpi_raw, lat_idx, lon_idx, False)
        cape = extract_variable(cape_raw, lat_idx, lon_idx, False)
        uh = extract_variable(uh_raw, lat_idx, lon_idx, False)

        u10 = extract_variable(u10_raw, lat_idx, lon_idx, False)
        v10 = extract_variable(v10_raw, lat_idx, lon_idx, False)
        ws, wd_deg = wind_speed_direction(u10, v10)
        wd_card = np.vectorize(wind_dir_to_cardinal)(wd_deg)

        capoluoghi_dati[city] = {
            't2m': np.round(t2m, 1),
            'rh2m': np.round(rh2m, 1),
            'tp': np.round(tp, 1),
            'clct': np.round(clct, 1),
            'clcl': np.round(clcl, 1),
            'clcm': np.round(clcm, 1),
            'clch': np.round(clch, 1),
            'lpi': np.round(lpi, 1),
            'cape': np.round(cape, 1),
            'uh': np.round(uh, 1),
            'pmsl': np.round(pmsl, 1),
            'wind_speed': np.round(ws, 1),
            'wind_dir_cardinal': wd_card,
            'tw': np.round(wet_bulb_celsius(t2m, rh2m), 1),
            'lat': lat_c,
            'lon': lon_c,
            'alt_model': round(hs_loc, 1),
            'alt_real': alt_c
        }

    return capoluoghi_dati

# === MAIN ===
if __name__ == '__main__':
    now = datetime.utcnow()
    if now.hour < 2:
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

# %% BOLLETTINO GIORNALIERO

print("Inizia creazione bollettino nazionale giornaliero...")
data_dir = os.path.join(os.getcwd(), "data")
icone_dir = os.path.join(os.getcwd(), "icons2")
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

capoluoghi_regione = {
    'Torino', 'Aosta', 'Milano', 'Trento', 'Venezia', 'Trieste', 'Genova', 'Bologna', 'Firenze', 'Perugia',
    'Ancona', 'Roma', 'L Aquila', 'Campobasso', 'Napoli', 'Potenza', 'Bari', 'Catanzaro', 'Palermo', 'Cagliari'
}
localita_extra = {
    "Bolzano", 'Sassari', 'Catania', 'Lecce', 'Elba', 'Foggia', 'Livigno',
    'Mar_Ionio', 'Mar_Tirreno', 'Mar_Adriatico', 'Formazza', 'Pantelleria', "Cortina d Ampezzo", "Parma", "Rimini", "Pescara", "Olbia", "Messina"
}
localita_interessate = capoluoghi_regione.union(localita_extra)
tz_italy = ZoneInfo('Europe/Rome')
satellite_path = os.path.join(icone_dir, "satellite.tif")

pickle_path = os.path.join(data_dir, 'capoluoghi_dati.pkl')
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)
    capoluoghi_dati = data['capoluoghi_dati']
    run_datetime_utc = data['run_datetime_utc']

run_folder = run_datetime_utc.strftime('%Y_%m_%d')
run_output_dir = os.path.join(output_dir, run_folder)
os.makedirs(run_output_dir, exist_ok=True)

offset_icone = {
    'Bologna': (0.2, 0.05), 'Firenze': (-0.1,0), 'Genova': (0,0.05), 'Torino': (-0.25, 0.1),
    "Bolzano": (0.1,0.1), 'Livigno': (-0.05,-0.1), 'Campobasso': (0.15,0.05), 'Mare_Adriatico': (0,0.4),
    'Elba': (0,-0.1), 'Trieste': (0.05,-0.1), 'Foggia': (0.25, 0.15), 'L Aquila': (0,0.05),
    'Cortina d Ampezzo': (0.25,0.35), 'Trento': (-0.25,-0.05), 'Parma': (-0.05,0),
    'Rimini': (0.2,0), 'Pescara': (0.2, 0.15), 'Perugia': (-0.05, -0.05), 'Reggio Calabria': (0.15, 0.1), 'Messina': (0.05, 0.05), 'Olbia': (0, 0.1), 'Catania': (-0.05, -0.1)
}

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
        
    all_land_union = unary_union(all_land)
    mask_geom = all_land_union.difference(italy_geom)
    
    def shapely_to_pathpatch(shapely_geom, **kwargs):
        if shapely_geom.geom_type == 'Polygon':
            verts, codes = [], []
            exterior = shapely_geom.exterior
            verts += list(exterior.coords)
            codes += [Path.MOVETO] + [Path.LINETO]*(len(exterior.coords)-2) + [Path.CLOSEPOLY]
            for interior in shapely_geom.interiors:
                verts += list(interior.coords)
                codes += [Path.MOVETO] + [Path.LINETO]*(len(interior.coords)-2) + [Path.CLOSEPOLY]
            return PathPatch(Path(verts, codes), **kwargs)
        elif shapely_geom.geom_type == 'MultiPolygon':
            return [shapely_to_pathpatch(part, **kwargs) for part in shapely_geom.geoms]
        return None

    patch = shapely_to_pathpatch(mask_geom, facecolor=to_rgba('white', 0.5), edgecolor='none', transform=ccrs.PlateCarree())
    if isinstance(patch, list):
        for p in patch: ax.add_patch(p)
    else:
        ax.add_patch(patch)

# MAPPA INDICI ORARI CON I GIORNI LOCALI REALI
n_time = len(capoluoghi_dati[next(iter(capoluoghi_dati))]['t2m'])
days_map = {}
for k in range(n_time):
    dt_utc_h = run_datetime_utc + timedelta(hours=k)
    loc_h = dt_utc_h.astimezone(tz_italy)
    day_str = loc_h.strftime("%Y-%m-%d")
    if day_str not in days_map:
        days_map[day_str] = []
    days_map[day_str].append(k)

valid_days = sorted(days_map.keys())

# Rimuovo il primo giorno e l'ultimo se il run è alle 12
if run_datetime_utc.hour == 12:
    # RUN 12 UTC: Salta "oggi" (indice 0) e prendi solo domani e dopodomani (indici 1 e 2)
    valid_days = valid_days[1:3]
else:
    # RUN 00 UTC: Prendi "oggi" (indice 0) e "domani" (indice 1) scartando la coda
    valid_days = valid_days[0:2]

# Ciclo dei giorni all'inverso per generare i bollettini
for day_str in sorted(valid_days, reverse=True):
    idxs = days_map[day_str]
    
    start_loc = run_datetime_utc.astimezone(tz_italy).replace(
        year=int(day_str[0:4]), month=int(day_str[5:7]), day=int(day_str[8:10]),
        hour=12, minute=0, second=0
    )
    
    fig = plt.figure(figsize=(12, 12))
    fig.patch.set_facecolor('#157acc')
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([5.75, 19.25, 35.5, 47.5], crs=ccrs.PlateCarree())
    
    with rasterio.open(satellite_path) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        ax.imshow(img, origin='upper', extent=extent, transform=ccrs.PlateCarree())
        
    schiarisci_fuori_italia(ax)
    ax.coastlines(resolution='10m', zorder=0)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), zorder=0)

    for city, dati in capoluoghi_dati.items():
        if city not in localita_interessate: continue
        
        lat, lon = dati['lat'], dati['lon']
        offset_lat, offset_lon = offset_icone.get(city, (0, 0))
        lat = lat + offset_lat
        lon = lon + offset_lon
        
        t2m = dati['t2m'][idxs]
        t2m_min = np.min(t2m)
        t2m_max = np.max(t2m)
        
        clct_avg = np.mean(dati['clct'][idxs])
        clcl_avg = np.mean(dati['clcl'][idxs])
        clcm_avg = np.mean(dati['clcm'][idxs])
        clch_avg = np.mean(dati['clch'][idxs])
        tp_sum = np.sum(dati['tp'][idxs])
        wind_speed_avg = np.mean(dati['wind_speed'][idxs])
        
        wd_list = [dati['wind_dir_cardinal'][i] for i in idxs]
        wind_dir_avg = Counter(wd_list).most_common(1)[0][0]

        seas, thr = get_season_precise(start_loc)

        recs = []
        for i in idxs:
            dt_utc_h = run_datetime_utc + timedelta(hours=i)
            loc_h = dt_utc_h.astimezone(tz_italy)
            wtxt = classify_weather_hourly(
                dati['t2m'][i], dati['rh2m'][i], dati['clct'][i],
                dati['clcl'][i], dati['clcm'][i], dati['clch'][i],
                dati['tp'][i], dati['wind_speed'][i],
                dati['lpi'][i], dati['cape'][i], dati['uh'][i],
                seas, thr
            )
            recs.append({"h": loc_h.hour, "w": wtxt})
        
        wdaily = classify_daily_weather(recs, clct_avg, clcl_avg, clcm_avg, clch_avg, tp_sum, seas, thr)

        if city.startswith("Mar_"):
            if wind_speed_avg < 10: nome_icona = "mare_1.png"
            elif wind_speed_avg < 20: nome_icona = "mare_2.png"
            else: nome_icona = "mare_3.png"
        else:
            nome_icona = f"{wdaily.lower().replace(' ', '_')}.png"

        path_icona = os.path.join(icone_dir, nome_icona)
        if os.path.exists(path_icona):
            img_icon = imread(path_icona)
            ab = AnnotationBbox(OffsetImage(img_icon, zoom=0.035), (lon, lat), frameon=False, transform=ccrs.PlateCarree())
            ax.add_artist(ab)
        else:
            print(f"Icona mancante: {nome_icona}")

        if city.startswith("Mar_"):
            box_x = lon - 0.35
            box_y = lat - 0.35
            text_y = box_y + 0.075
        else:
            box_x = lon - 0.3
            box_y = lat - 0.5
            text_y = box_y + 0.075
            
            ax.add_patch(plt.Rectangle(
                (box_x, box_y), 0.7, 0.15,
                transform=ccrs.PlateCarree(),
                color='white', alpha=0.7, zorder=10
            ))
            
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
        imagebox = OffsetImage(logo_img, zoom=0.07)
        ab_logo = AnnotationBbox(imagebox, (0.18, 0.05), xycoords=ax.transAxes,
                                 frameon=False, box_alignment=(1, 1), zorder=15)
        ax.add_artist(ab_logo)

    run_hour_str = run_datetime_utc.strftime('%H')
    locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')
    line1 = f"Bollettino Italia - {start_loc.strftime('%A %d/%m/%Y')}"
    line2 = f"ICON 2I - run: {run_datetime_utc.strftime('%d/%m/%Y %H')}"
    ax.text(0.5, 1.05, line1, ha='center', va='bottom', fontsize=16, weight='bold', transform=ax.transAxes)
    ax.text(0.5, 1.03, line2, ha='center', va='top', fontsize=12, transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(f"{run_output_dir}/{start_loc.strftime('%d-%m-%Y')}_{run_hour_str}.png", dpi=120, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

print("Fine creazione bollettino giornaliero.")

# -------------------------------------------------------------------------
# CARICAMENTO IMMAGINI SU CLOUDFLARE R2
# -------------------------------------------------------------------------
print("Procedura R2 iniziata.")

R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
BUCKET_NAME = "bollettini"

if R2_ACCESS_KEY and R2_SECRET_KEY and R2_ENDPOINT:
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY
        )
        
        tz_it = ZoneInfo("Europe/Rome")
        run_h_str = run_datetime_utc.strftime('%H')

        for day_str in valid_days:
            start_loc = run_datetime_utc.astimezone(tz_it).replace(
                year=int(day_str[0:4]), month=int(day_str[5:7]), day=int(day_str[8:10]),
                hour=12, minute=0, second=0
            )

            filename_base = f"{start_loc.strftime('%d-%m-%Y')}_{run_h_str}"
            png_filename = f"{filename_base}.png"
            png_path = os.path.join(run_output_dir, png_filename)

            if os.path.exists(png_path):
                try:
                    with Image.open(png_path) as img:
                        with io.BytesIO() as output_buffer:
                            img.save(output_buffer, format="WEBP", quality=75, dpi=(120, 120))
                            output_buffer.seek(0)

                            webp_filename = f"{filename_base}.webp"
                            s3_client.upload_fileobj(
                                output_buffer,
                                BUCKET_NAME,
                                webp_filename,
                                ExtraArgs={'ContentType': 'image/webp'}
                            )
                            print(f" -> Upload completato: {webp_filename}")
                except Exception as e:
                    print(f"Errore processamento/upload {png_filename}: {e}")
            else:
                print(f"Attenzione: File locale non trovato per upload: {png_path}")
                
        print("Procedura R2 terminata.")
    except Exception as e:
        print(f"Errore generale connessione R2: {e}")
else:
    print("Credenziali R2 (R2_ACCESS_KEY, R2_SECRET_KEY, R2_ENDPOINT) mancanti. Upload saltato.")
