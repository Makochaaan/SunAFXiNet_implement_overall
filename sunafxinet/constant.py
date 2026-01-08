# --- Effect Definitions ---
EFFECT_TYPES = ['Distortion', 'Chorus', 'Delay', 'Reverb']
NUM_EFFECTS = len(EFFECT_TYPES)

# --- Parameter Dims ---
PARAM_DIMS = {
    'Distortion': 1,
    'Chorus': 3,
    'Delay': 3,
    'Reverb': 4,
}

# --- Parameter Names ---
EFFECT_PARAM_NAMES = {
    'Distortion': ['drive_db'],
    'Chorus': ['rate_hz', 'depth', 'mix'],
    'Delay': ['delay_seconds', 'feedback', 'mix'],
    'Reverb': ['room_size', 'damping', 'wet_level', 'dry_level']
}

PARAM_RANGES = {
    'Distortion': {'drive_db': (10.0, 40.0)},
    'Chorus': {'rate_hz': (0.5, 5.0), 'depth': (0.2, 0.8), 'mix': (0.1, 0.5)},
    'Delay': {'delay_seconds': (0.1, 0.8), 'feedback': (0.1, 0.6), 'mix': (0.1, 0.5)},
    'Reverb': {'room_size': (0.1, 0.9), 'damping': (0.1, 0.9), 'wet_level': (0.1, 0.5), 'dry_level': (0.5, 0.9)}
}

EFFECT_MAP = {name: i for i, name in enumerate(EFFECT_TYPES)}

INV_EFFECT_MAP = {i: name for name, i in EFFECT_MAP.items()}

# --- HDEMUCS Config (Paper exact) ---
HDEMUCS_CONFIG = {
    'audio_channels': 1,
    'channels': 32,
    'growth': 2,
    'nfft': 8192,
    'cac': True,
    'depth': 6,
    'rewrite': True,
    'dconv_mode': 3,
    't_layers': 3,
    't_heads': 8,
    'segment': 10.0,
    'use_train_segment': False  # 固定長セグメントを強制
}

# --- Training ---
SAMPLE_RATE = 48000
BATCH_SIZE = 8
LR_STAGE1 = 5e-5
LR_STAGE2 = 5e-5
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 20
LAMBDA_STFT = 0.05
