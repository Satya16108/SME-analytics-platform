# ============================================================
#  DECISION INTELLIGENCE PLATFORM — Configuration
#  Add new sectors to SECTOR_REGISTRY to scale the platform
# ============================================================

COLORS = {
    "navy":       "#1B3A6B",
    "teal":       "#0E7C86",
    "orange":     "#E07B39",
    "red":        "#C0392B",
    "green":      "#27AE60",
    "yellow":     "#F39C12",
    "purple":     "#8E44AD",
    "light_bg":   "#F5F7FA",
    "white":      "#FFFFFF",
    "text_dark":  "#2C3E50",
    "text_mid":   "#5D6D7E",
    "border":     "#E8ECF0",
}

# ── Chart palette (used in Plotly figures) ───────────────────
CHART_COLORS = [
    COLORS["navy"], COLORS["teal"], COLORS["orange"],
    COLORS["green"], COLORS["purple"], COLORS["yellow"],
]

PLATFORM_CONFIG = {
    "name":     "Anviksha",
    "subtitle": "AI-Powered Decision Intelligence for Indian SMEs",
    "version":  "v1.0.0",
}

# ── Sector Registry ──────────────────────────────────────────
# To add a new sector:
#   1. Create  sectors/<your_sector>.py  with a class inheriting BaseSector
#   2. Add an entry below  — nothing else needs to change
SECTOR_REGISTRY = {
    "manufacturing": {
        "display_name": "⚙️ Manufacturing",
        "subtitle":     "Predictive Maintenance Intelligence",
        "description":  "Predict machine failures before they happen using real-time sensor analytics.",
        "core_problem": "Unplanned downtime: 20–35% production capacity lost per year",
        "ml_model":     "Random Forest Classifier",
        "accent":       COLORS["navy"],
        "module":       "sectors.manufacturing",
        "class_name":   "PredictiveMaintenanceSector",
    },
    "retail": {
        "display_name": "🛒 Retail & E-Commerce",
        "subtitle":     "Demand Forecasting Intelligence",
        "description":  "Forecast SKU-level demand to eliminate dead stock and prevent stockouts.",
        "core_problem": "15–25% of inventory becomes dead stock, locking working capital",
        "ml_model":     "XGBoost Regressor",
        "accent":       COLORS["teal"],
        "module":       "sectors.retail",
        "class_name":   "DemandForecastingSector",
    },
    "agrifood": {
        "display_name": "🌾 Agri-Food Processing",
        "subtitle":     "Crop Price Forecasting Intelligence",
        "description":  "Forecast raw material prices to optimise procurement timing and hedge risk.",
        "core_problem": "Raw material price volatility of 20–40% per season squeezes margins",
        "ml_model":     "XGBoost with Seasonal Features",
        "accent":       COLORS["orange"],
        "module":       "sectors.agrifood",
        "class_name":   "PriceForecastingSector",
    },
    "it_services": {
        "display_name": "💻 IT & Tech Services",
        "subtitle":     "Employee Attrition Prediction",
        "description":  "Identify employees at risk of leaving and take pre-emptive retention actions.",
        "core_problem": "18–35% annual attrition — ₹3–8L replacement cost per head",
        "ml_model":     "Gradient Boosting Classifier",
        "accent":       COLORS["purple"],
        "module":       "sectors.it_services",
        "class_name":   "AttritionPredictionSector",
    },
    "healthcare": {
        "display_name": "🏥 Healthcare & Pharma",
        "subtitle":     "Batch Failure Prediction",
        "description":  "Predict pharmaceutical batch failures from in-process parameters before rejection.",
        "core_problem": "3–8% batch failure rate causing lakhs in rework and raw material loss",
        "ml_model":     "Random Forest Classifier",
        "accent":       COLORS["green"],
        "module":       "sectors.healthcare",
        "class_name":   "BatchFailureSector",
    },
}
