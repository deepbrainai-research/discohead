from modules.audio_encoder import AudioEncoder

params = {
    'fps': 25,
    'samplerate': 22050,
    'weight_path': './weight/grid.pt',
    'model_params': {
        'head_predictor': {
            'num_affines': 1,
            'using_scale': False
        },
        'generator': {
            'num_affines': 1,
            'num_residual_mod_blocks': 6,
            'using_gaussian': False
        },
        'audio_encoder': AudioEncoder
    }
}
