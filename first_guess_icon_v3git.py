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
    # ... mantieni qui la lista completa
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
    hsurf_raw = data['HSURF']['h'].values

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
