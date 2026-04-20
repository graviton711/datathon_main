# Feature names and constant mappings

TARGET_COL = "Revenue"
DATE_COL = "Date"

CATEGORICAL_COLS = [
    "category",
    "segment",
    "size",
    "color",
    "city",
    "region",
    "gender",
    "age_group",
    "acquisition_channel",
    "traffic_source",
]

NUMERICAL_COLS = [
    "price",
    "cogs",
    "quantity",
    "stock_on_hand",
    "page_views",
    "sessions",
]

# Regional mappings if needed
REGION_MAPPING = {
    "West": 0,
    "Central": 1,
    "East": 2
}
