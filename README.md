# urban-heat-lst

**Open-Source Urban Heat Island (UHI) & Land Surface Temperature (LST) Analysis Pipeline**

`urban-heat-lst` is a modular, command-line Python tool that automates the detection, classification, and mapping of Urban Heat Islands (UHI) using **Landsat 8 & 9 Collection 2 Level-2** satellite imagery streamed directly from the **Microsoft Planetary Computer STAC API**.

---

## Sample Results — Antalya Central Districts (15 August 2024)

| Land Surface Temperature (LST °C) | 5-Class Urban Heat Island (UHI) |
| :---: | :---: |
| ![LST Map](docs/images/lst_map.png) | ![UHI Classified Map](docs/images/uhi_classified_map.png) |

| Normalized Difference Vegetation Index (NDVI) | LST vs. NDVI Correlation ($r = -0.47$) |
| :---: | :---: |
| ![NDVI Map](docs/images/ndvi_map.png) | ![LST vs NDVI Correlation](docs/images/lst_ndvi_correlation.png) |

---

## Key Features

1. **Configuration-Driven Workflow (`config.yaml`)**: Reads Area of Interest (GeoJSON), date range, seasonal month filter (`season_months`), maximum cloud cover threshold, configurable `QA_PIXEL` bit list (`qa_mask_bits`), and minimum hotspot area (`min_area_ha`).
2. **Automated STAC Scene Selection with Seasonal Filtering**: Searches `landsat-c2-l2` via `pystac-client` and `planetary-computer`, filters by target season months (default `[6, 7, 8]` for June–August), verifies spatial AOI coverage, and selects the least-cloudy scene.
3. **Configurable Cloud, Shadow, Snow & Water Bitmasking (`QA_PIXEL`)**: Decodes Landsat Collection 2 `QA_PIXEL` bit flags from `qa_mask_bits` (default `[0, 1, 2, 3, 4, 5, 7]` corresponding to Fill, Dilated Cloud, Cirrus, Cloud, Cloud Shadow, Snow [Bit 5], and Water [Bit 7]) before clipping to the AOI.
4. **Physical LST (°C) & NDVI Retrieval**:
   - Converts Thermal Infrared (`ST_B10` / `lwir11`) digital numbers to Kelvin and Celsius ($\text{LST}_{^\circ\text{C}} = (\text{DN} \times 0.00341802 + 149.0) - 273.15$).
   - Scales Surface Reflectance (`SR_B4` Red & `SR_B5` NIR) and computes the Normalized Difference Vegetation Index ($\text{NDVI}$).
5. **5-Class UHI Classification & Sieved Hotspot Vectorization**:
   - Computes spatial Mean ($\mu$) and Standard Deviation ($\sigma$) strictly from valid post-masked pixels and classifies LST into 5 thermal zones:
     - **Class 1 (Very Low)**: $< \mu - 1.5\sigma$
     - **Class 2 (Low)**: $[\mu - 1.5\sigma, \mu - 0.5\sigma)$
     - **Class 3 (Moderate)**: $[\mu - 0.5\sigma, \mu + 0.5\sigma)$
     - **Class 4 (High)**: $[\mu + 0.5\sigma, \mu + 1.5\sigma)$
     - **Class 5 (Very High / Hotspot)**: $\ge \mu + 1.5\sigma$
   - Applies `rasterio.features.sieve` prior to vectorization to remove noise patches smaller than `min_area_ha` (default `0.5` ha), exporting clean hotspot polygons to `outputs/hotspots.gpkg`.
6. **Ready-to-Use Geospatial & Visual Outputs**:
   - `outputs/lst_celsius.tif` — Continuous LST GeoTIFF (°C, Float32)
   - `outputs/lst_classified.tif` — 5-class UHI GeoTIFF (UInt8)
   - `outputs/hotspots.gpkg` — Vector polygons of extreme UHI hotspots
   - `outputs/statistics.csv` — Summary metrics (`min`, `max`, `mean`, `std`, `median`, `mean_ndvi`, and Pearson `lst_ndvi_corr`)
   - `outputs/maps/*.png` — High-resolution maps (`lst_map.png`, `uhi_classified_map.png`, `ndvi_map.png`, `lst_ndvi_correlation.png`)

---

## Project Structure

```text
urban-heat-lst/
├── config.yaml                 # Pipeline configuration
├── data/
│   └── antalya_center.geojson  # Sample AOI: Antalya central districts (Muratpaşa, Kepez, Konyaaltı, Aksu, Döşemealtı)
├── docs/
│   └── images/                 # Sample PNG maps displayed in README
├── urban_heat_lst/
│   ├── __init__.py
│   ├── config.py               # Configuration & AOI GeoJSON loader
│   ├── stac_search.py          # Planetary Computer STAC API search & seasonal scene selection
│   ├── preprocessing.py        # COG windowed clipping & QA_PIXEL bitmasking
│   ├── indices.py              # ST_B10 to Celsius LST & SR_B4/B5 NDVI computation
│   ├── analysis.py             # 5-class Mean ± SD thresholding, sieve vectorization & statistics
│   └── visualization.py        # Matplotlib cartographic map & regression plot generation
├── tests/                      # Unit tests (pytest)
│   ├── test_preprocessing.py
│   ├── test_indices.py
│   └── test_analysis.py
├── run.py                      # Single-command CLI entry point
├── requirements.txt            # Python package dependencies
└── README.md
```

---

## Installation & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Parameters (`config.yaml`)

```yaml
aoi_path: "data/antalya_center.geojson"
date_range: "2024-01-01/2024-12-31"
season_months: [6, 7, 8]          # Default: June-August (summer months)
max_cloud_cover: 15.0
qa_mask_bits: [0, 1, 2, 3, 4, 5, 7] # 0:Fill, 1:DilatedCloud, 2:Cirrus, 3:Cloud, 4:Shadow, 5:Snow, 7:Water
hotspot_min_class: 5
min_area_ha: 0.5                  # Minimum hotspot area in hectares (via rasterio.features.sieve)
output_dir: "outputs"
```

### 3. Run the Full Pipeline

```bash
python run.py --config config.yaml
```

### 4. Run Unit Tests

```bash
pytest tests/
```

---

## Türkçe Özet (Turkish Summary)

**`urban-heat-lst`**, uydu görüntüleri kullanarak şehirlerdeki **Kentsel Isı Adası (Urban Heat Island - UHI)** etkisini otomatik olarak analiz eden açık kaynaklı bir Python aracıdır.

### Nasıl Çalışır?
1. **Yapılandırma (`config.yaml`)**: Çalışma alanı sınırlarını (`data/antalya_center.geojson` - Antalya merkez ilçeleri), tarih aralığını, mevsim filtresini (`season_months`, varsayılan: Haziran–Ağustos `[6, 7, 8]`), maksimum bulutluluk oranını, maskelenecek `QA_PIXEL` bit listesini (`qa_mask_bits: [0, 1, 2, 3, 4, 5, 7]`) ve minimum sıcak nokta alanını (`min_area_ha: 0.5`) okur.
2. **Uydu Görüntüsü Arama (`stac_search.py`)**: Microsoft Planetary Computer STAC API üzerinden **Landsat 8/9 Collection 2 Level-2** arşivini tarar, mevsim filtresini uygular ve çalışma alanını kapsayan en az bulutlu görüntüyü seçer.
3. **Ön İşleme ve Maskeleme (`preprocessing.py`)**: `QA_PIXEL` kalite bandında `qa_mask_bits` listesinde tanımlı bitleri (Bit 0–4 bulut/gölge, Bit 5 kar, Bit 7 su) maskeler ve bantları çalışma alanına göre kırpar.
4. **LST ve NDVI Hesaplama (`indices.py`)**: `ST_B10` termal bandını ölçek katsayılarıyla **Celsius (°C)** cinsinden Yer Yüzey Sıcaklığına (LST) dönüştürür; Kırmızı (`SR_B4`) ve Yakın Kızılötesi (`SR_B5`) bantlarından **NDVI** hesaplar.
5. **Sınıflandırma ve Sıcak Nokta Vektörleştirme (`analysis.py`)**: Maskeleme sonrası geçerli piksellerden ortalama ($\mu$) ve standart sapmayı ($\sigma$) hesaplayarak yüzey sıcaklığını 5 sınıfa ayırır. Vektörleştirme öncesinde `rasterio.features.sieve` ile `min_area_ha` (varsayılan `0.5` ha) altındaki küçük piksel kümelerini temizler ve sıcak noktaları `outputs/hotspots.gpkg` olarak dışa aktarır.
6. **Çıktılar (`outputs/`)**:
   - `lst_celsius.tif` ve `lst_classified.tif` (GeoTIFF rasterlar)
   - `hotspots.gpkg` (Sıcak nokta vektör poligonları)
   - `statistics.csv` (Min, maks, ortalama sıcaklık ve LST–NDVI korelasyonu)
   - `maps/*.png` (LST, NDVI, 5 sınıflı ısı adası haritası ve LST-NDVI korelasyon grafiği)
