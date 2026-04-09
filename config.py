# config.py

from datetime import date

# Years supported for legacy video/monthly framing (derived from map dates in the UI)
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Earliest date allowed for Dynamic World day mosaics (collection coverage)
DW_MIN_DATE = date(2018, 1, 1)

# Default location (used when city is empty or geocoding fails)
LOCATION_LAT = 24.4539   # Abu Dhabi latitude
LOCATION_LON = 54.3773   # Abu Dhabi longitude
LOCATION_NAME = "Abu Dhabi"

# Dynamic World class labels
CLASS_LABELS = [
    "Water",
    "Trees",
    "Grass",
    "Flooded vegetation",
    "Crops",
    "Shrub & scrub",
    "Built area",
    "Bare ground",
    "Snow & ice",
]

# Dynamic World class palette (hex colors as strings, without "#")
CLASS_PALETTE = [
    "419bdf",  # Water
    "397d49",  # Trees
    "88b053",  # Grass
    "7a87c6",  # Flooded vegetation
    "e49635",  # Crops
    "dfc35a",  # Shrub & scrub
    "c4281b",  # Built area
    "a59b8f",  # Bare ground
    "b39fe1",  # Snow & ice
]
