from modules.audio_encoder_bn import AudioEncoder

params = {
    'fps': 29.97,
    'samplerate': 22050,
    'weight_path': './weight/koeba.pt',
    'model_params': {
        'head_predictor': {
            'num_affines': 1,
            'using_scale': True
        },
        'generator': {
            'num_affines': 1,
            'num_residual_mod_blocks': 6,
            'using_gaussian': False
        },
        'audio_encoder': AudioEncoder
    }
}
