# config/settings.py
# Configuration constants and settings

# Embedding Model Configuration
EMBED_MODEL = "all-MiniLM-L6-v2"

# Greeting Service Configuration
GREETING_INTENT = "greeting"  # The intent name for greetings
GRATITUDE_INTENT = "gratitude"  # The intent name for gratitude
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to consider as greeting/gratitude (updated from 0.6)

# Hybrid Routing Thresholds
FAQ_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for FAQ semantic search
ML_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for ML intent classification routing

# Per-Domain Config Directories
GREETINGS_DIR = "configs/greetings"      # Per-domain greeting configs
PERSONAS_DIR = "configs/personas"        # Per-domain persona configs
INTENTS_DIR = "configs/intents"          # Per-domain intent configs

# Legacy path (for migration detection)
LEGACY_GREETING_CONFIG_PATH = "greeting_config/greeting_config.json"

# Default Domain
DEFAULT_DOMAIN = "algobridge"

