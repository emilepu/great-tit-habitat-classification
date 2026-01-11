import ee
import pandas as pd
import json

#code to assign each recoring a habitat distribution (of 300 pixels)
#based on the coordinates and year of recording


# GEE authentication
proj_id = # replace with your GEE project ID
ee.Authenticate()
ee.Initialize(project=proj_id)

df = pd.read_csv("/Parus_major_metadata.csv") # replace with CSV file path
BUFFER_RADIUS = 300  

df = df.dropna(subset=['year']) # ensure all instances have year
df['year'] = df['year'].astype(int)



# create feature collection from dataframe
features = [
    ee.Feature(ee.Geometry.Point([row['lon'], row['lat']]), {
        'id': row['id'],
        'year': row['year']
    })
    for _, row in df.iterrows()
]
bird_fc = ee.FeatureCollection(features)

# --- class lookupf for datasets ---
modis_lookup = {
    0: "Water",
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up",
    14: "Cropland/Natural Vegetation Mosaics",
    15: "Snow and Ice",
    16: "Barren or Sparsely Vegetated",
    17: "Unclassified"
}
dw_lookup = {
    0: "Water",
    1: "Trees",
    2: "Grass",
    3: "Flooded Vegetation",
    4: "Crops",
    5: "Shrub & Scrub",
    6: "Built Area",
    7: "Bare Ground",
    8: "Snow & Ice"
}


def modis_to_habitat_category(lc_value): # map MODIS class to a habitat label
    if lc_value == 13:
        return 'Urban'
    elif lc_value <= 5:
        return 'Forest'
    elif lc_value in [0, 15]:
        return 'Misc'
    else:
        return 'Open'

def dw_to_habitat_category(lc_value): #map Dynamic Wordl class to a habitat label
    if lc_value == 6:
        return 'Urban'
    elif lc_value in [1, 5]:
        return 'Forest'
    elif lc_value in [0, 8]:
        return 'Misc'
    else:
        return 'Open'

# -------------------------------- Section written with the assistance of ChatGPT --------------------------------
# -----------------------------
# Process histogram to habitat distribution
# -----------------------------
def process_modis_histogram(histogram_dict):
    """Convert MODIS histogram to habitat distribution"""
    habitat_counts = {'Forest': 0, 'Urban': 0, 'Open': 0, 'Misc': 0}
    total = 0
    
    for lc_str, count in histogram_dict.items():
        try:
            lc_value = int(float(lc_str))
            habitat = modis_to_habitat_category(lc_value)
            habitat_counts[habitat] += count
            total += count
        except:
            continue
    
    # Convert to probabilities
    if total > 0:
        habitat_dist = {k: v/total for k, v in habitat_counts.items()}
        dominant_habitat = max(habitat_dist, key=habitat_dist.get)
    else:
        habitat_dist = {'Forest': 0, 'Urban': 0, 'Open': 0, 'Misc': 0}
        dominant_habitat = 'Unknown'
    
    return habitat_dist, dominant_habitat

def process_dw_histogram(histogram_dict):
    """Convert Dynamic World histogram to habitat distribution"""
    habitat_counts = {'Forest': 0, 'Urban': 0, 'Open': 0, 'Misc': 0}
    total = 0
    
    for lc_str, count in histogram_dict.items():
        try:
            lc_value = int(float(lc_str))
            habitat = dw_to_habitat_category(lc_value)
            habitat_counts[habitat] += count
            total += count
        except:
            continue
    
    # Convert to probabilities
    if total > 0:
        habitat_dist = {k: v/total for k, v in habitat_counts.items()}
        dominant_habitat = max(habitat_dist, key=habitat_dist.get)
    else:
        habitat_dist = {'Forest': 0, 'Urban': 0, 'Open': 0, 'Misc': 0}
        dominant_habitat = 'Unknown'
    
    return habitat_dist, dominant_habitat

# -----------------------------
# Sample points by year with buffer
# -----------------------------
years = sorted(df['year'].unique())
results = []

for year in years:
    year_int = int(year)
    print(f"Processing year {year_int}...")
    
    points_year = bird_fc.filter(ee.Filter.eq('year', year_int))
    
    if 2001 <= year_int <= 2022:
        # --- MODIS Land Cover ---
        lc_collection = ee.ImageCollection("MODIS/061/MCD12Q1") \
            .filter(ee.Filter.calendarRange(year_int, year_int, 'year'))
        
        collection_size = lc_collection.size().getInfo()
        
        if collection_size > 0:
            lc_img = lc_collection.first().select('LC_Type1')
            
            # Create buffered points
            buffered = points_year.map(lambda f: f.buffer(BUFFER_RADIUS))
            
            # Sample with frequency histogram reducer
            sampled = lc_img.reduceRegions(
                collection=buffered,
                reducer=ee.Reducer.frequencyHistogram(),
                scale=500
            )
            
            # Get the results and process locally
            sampled_list = sampled.getInfo()['features']
            
            for feature in sampled_list:
                props = feature['properties']
                histogram = props.get('histogram', {})
                
                habitat_dist, dominant_habitat = process_modis_histogram(histogram)
                
                results.append({
                    'id': props['id'],
                    'year': props['year'],
                    'habitat': dominant_habitat,
                    'habitat_forest': habitat_dist['Forest'],
                    'habitat_urban': habitat_dist['Urban'],
                    'habitat_open': habitat_dist['Open'],
                    'habitat_misc': habitat_dist['Misc'],
                    'dataset': 'MODIS'
                })
        else:
            print(f"Warning: No MODIS data for year {year_int}")
            # Add rows with no data
            for _, row in df[df['year'] == year_int].iterrows():
                results.append({
                    'id': row['id'],
                    'year': year_int,
                    'habitat': 'No Data',
                    'habitat_forest': 0,
                    'habitat_urban': 0,
                    'habitat_open': 0,
                    'habitat_misc': 0,
                    'dataset': 'None'
                })
    
    elif year_int >= 2023:
        # --- Dynamic World Land Cover ---
        dw_collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
            .filterDate(f"{year_int}-01-01", f"{year_int}-12-31")
        
        collection_size = dw_collection.size().getInfo()
        
        if collection_size > 0:
            dw_img = dw_collection.mode().select('label')
            
            # Create buffered points
            buffered = points_year.map(lambda f: f.buffer(BUFFER_RADIUS))
            
            # Sample with frequency histogram reducer
            sampled = dw_img.reduceRegions(
                collection=buffered,
                reducer=ee.Reducer.frequencyHistogram(),
                scale=10
            )
            
            # Get the results and process locally
            sampled_list = sampled.getInfo()['features']
            
            for feature in sampled_list:
                props = feature['properties']
                histogram = props.get('histogram', {})
                
                habitat_dist, dominant_habitat = process_dw_histogram(histogram)
                
                results.append({
                    'id': props['id'],
                    'year': props['year'],
                    'habitat': dominant_habitat,
                    'habitat_forest': habitat_dist['Forest'],
                    'habitat_urban': habitat_dist['Urban'],
                    'habitat_open': habitat_dist['Open'],
                    'habitat_misc': habitat_dist['Misc'],
                    'dataset': 'DynamicWorld'
                })
        else:
            print(f"Warning: No Dynamic World data for year {year_int}")
            # Add rows with no data
            for _, row in df[df['year'] == year_int].iterrows():
                results.append({
                    'id': row['id'],
                    'year': year_int,
                    'habitat': 'No Data',
                    'habitat_forest': 0,
                    'habitat_urban': 0,
                    'habitat_open': 0,
                    'habitat_misc': 0,
                    'dataset': 'None'
                })
    
    else:
        # --- Before MODIS coverage ---
        print(f"Info: Year {year_int} is before MODIS coverage (2001-2022)")
        for _, row in df[df['year'] == year_int].iterrows():
            results.append({
                'id': row['id'],
                'year': year_int,
                'habitat': 'Pre-MODIS',
                'habitat_forest': 0,
                'habitat_urban': 0,
                'habitat_open': 0,
                'habitat_misc': 0,
                'dataset': 'None'
            })

# -------------------------------------------------------------------------------------------------------



results= pd.DataFrame(results)
output_filename = 'habitat_metadata_soft_labels.csv'
results.to_csv(output_filename, index=False)

print(f"\n Done!")
print(f"total records: {len(results)}")