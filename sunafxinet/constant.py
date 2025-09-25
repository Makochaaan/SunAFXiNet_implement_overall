# ============================================================================
# Project-wide Configuration for SunAFXiNet
# ============================================================================

# --- 1. Effect Definitions ---
# The four effect types used in the project.
# This list is the single source of truth for the effect order.
# EFFECT_TYPES = ['Distortion', 'Chorus', 'Delay', 'Reverb']
# EFFECT_TYPES = ['Distortion', 'Reverb']
EFFECT_TYPES = ['Delay', 'Reverb']
# EFFECT_TYPES = ['Chorus', 'Reverb']

# --- Parameter Information ---
# Defines how many parameters each effect has.
PARAM_DIMS = {
    'Distortion': 1,
    'Chorus': 3,
    'Delay': 3,
    'Reverb': 4
}

EFFECT_MAP = {name: i for i, name in enumerate(EFFECT_TYPES)}

INV_EFFECT_MAP = {i: name for name, i in EFFECT_MAP.items()}

NUM_EFFECTS = len(EFFECT_TYPES)

# --- Parameter Ranges for Normalization ---
# Defines the (min, max) range for each parameter.
# Used for normalizing parameters to [0, 1] for training,
# and de-normalizing them back to their original range for evaluation.
PARAM_RANGES = {
    'Distortion': {'drive_db': (10.0, 40.0)},
    'Chorus': {'rate_hz': (0.5, 5.0), 'depth': (0.2, 0.8), 'mix': (0.1, 0.5)},
    'Delay': {'delay_seconds': (0.1, 0.8), 'feedback': (0.1, 0.6), 'mix': (0.1, 0.5)},
    'Reverb': {'room_size': (0.1, 0.9), 'damping': (0.1, 0.9), 'wet_level': (0.1, 0.5), 'dry_level': (0.5, 0.9)}
}

# --- Parameter Names for Pedalboard ---
# Defines the correct order of parameter names (keyword arguments)
# required by the `pedalboard` library.
EFFECT_PARAM_NAMES = {
    'Distortion': ['drive_db'],
    'Chorus': ['rate_hz', 'depth', 'mix'],
    'Delay': ['delay_seconds', 'feedback', 'mix'],
    'Reverb': ['room_size', 'damping', 'wet_level', 'dry_level']
}

# --- Hdemucs' Config ---
# Note: 'samplerate' is set dynamically in the training/evaluation scripts
HDEMUCS_CONFIG = {
    'audio_channels': 1, 'channels': 48, 'growth': 2, 'nfft': 4096,
    'cac': True, 'depth': 5, 'rewrite': True, 'dconv_mode': 3,
    't_layers': 4, 'segment': 10.0,
    't_heads': 8
}

# --- Training Hyperparameters ---
BATCH_SIZE = 4
LR_STAGE1, LR_STAGE2 = 5e-5, 5e-5
# EPOCHS_STAGE1, EPOCHS_STAGE2 = 400, 150
EPOCHS_STAGE1, EPOCHS_STAGE2 = 50, 40
LAMBDA_STFT = 0.05
SAMPLE_RATE = 48000
DATASET_DIR = '/home/depontes25/Desktop/Research/dataset/sunafxinet/wet_signal_2way/reb-cho/train'
