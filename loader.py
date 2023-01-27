import torch
import librosa
import numpy as np

from glob import glob
from os.path import join
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

from model import Model


def model_loader(config_dict):
    with torch.no_grad():
        model = Model(config_dict['model_params'])
    checkpoint = torch.load(config_dict['weight_path'])
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval().cuda()
    return model


def obama_demo1_data_loader(params):
    dataset_root = params['dataset_root']

    # load audio
    audio, _ = librosa.load(join(dataset_root, 'audio.wav'))

    # load features
    audio_features = np.load(join(dataset_root, 'audio_features.npy'))
    silence_feature = np.load(join(dataset_root, 'silence_feature.npy'))

    # parsing image paths
    src_path = join(dataset_root, 'src.jpg')
    masked_src_path = join(dataset_root, 'masked_src.png')
    head_driver_paths = sorted(glob(join(dataset_root, 'head_driver/*.jpg')))
    eye_driver_paths = sorted(glob(join(dataset_root, 'eye_driver/*.jpg')))
    lip_driver_paths = sorted(glob(join(dataset_root, 'lip_driver/*.jpg')))
    masked_driver_paths = sorted(glob(join(dataset_root, 'masked_driver/*.png')))

    # load images
    n_frames = len(audio_features)

    src = read_image(src_path, ImageReadMode.RGB)
    masked_src = read_image(masked_src_path, ImageReadMode.RGB)

    head_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)
    eye_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)
    lip_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)
    masked_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)

    for i in range(n_frames):
        head_driver[i] = read_image(head_driver_paths[i], ImageReadMode.RGB)
        eye_driver[i] = read_image(eye_driver_paths[i], ImageReadMode.RGB)
        lip_driver[i] = read_image(lip_driver_paths[i], ImageReadMode.RGB)
        masked_driver[i] = read_image(masked_driver_paths[i], ImageReadMode.RGB)

    data = {
        'n_frames': n_frames,
        'audio': torch.from_numpy(audio).view(1,-1), # (1, L)
        'audio_features': torch.from_numpy(audio_features), # (N, C, L)
        'silence_feature': torch.from_numpy(silence_feature), # (C, L)
        'src': src,
        'masked_src': masked_src,
        'head_driver': head_driver,
        'eye_driver': eye_driver,
        'lip_driver': lip_driver,
        'masked_driver': masked_driver
    }

    return data


def obama_demo2_data_loader(params):
    dataset_root = params['dataset_root']

    # load audio
    audio, _ = librosa.load(join(dataset_root, 'audio.wav'))

    # load features
    audio_features = np.load(join(dataset_root, 'audio_features.npy'))
    silence_feature = np.load(join(dataset_root, 'silence_feature.npy'))

    # parsing image paths
    src_path = join(dataset_root, 'src.jpg')
    masked_src_path = join(dataset_root, 'masked_src.png')
    driver_paths = sorted(glob(join(dataset_root, 'driver/*.jpg')))
    masked_driver_paths = sorted(glob(join(dataset_root, 'masked_driver/*.png')))

    # load images
    n_frames = len(audio_features)

    src = read_image(src_path, ImageReadMode.RGB)
    masked_src = read_image(masked_src_path, ImageReadMode.RGB)
    
    driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)
    masked_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)

    for i in range(n_frames):
        driver[i] = read_image(driver_paths[i], ImageReadMode.RGB)
        masked_driver[i] = read_image(masked_driver_paths[i], ImageReadMode.RGB)

    data = {
        'n_frames': n_frames,
        'audio': torch.from_numpy(audio).view(1,-1), # (1, L)
        'audio_features': torch.from_numpy(audio_features), # (N, C, L)
        'silence_feature': torch.from_numpy(silence_feature), # (C, L)
        'src': src,
        'masked_src': masked_src,
        'driver': driver,
        'masked_driver': masked_driver
    }

    return data


def grid_demo1_data_loader(params):
    dataset_root = params['dataset_root']

    # load audio
    audio, _ = librosa.load(join(dataset_root, 'audio.wav'))

    # load features
    audio_features = np.load(join(dataset_root, 'audio_features.npy'))

    # parsing image paths
    src1_path = join(dataset_root, 'src1.jpg')
    src2_path = join(dataset_root, 'src2.jpg')
    src3_path = join(dataset_root, 'src3.jpg')
    
    driver_paths = sorted(glob(join(dataset_root, 'driver/*.jpg')))
    masked_driver_paths = sorted(glob(join(dataset_root, 'masked_driver/*.png')))

    # load images
    n_frames = len(audio_features)

    src1 = read_image(src1_path, ImageReadMode.RGB)
    src2 = read_image(src2_path, ImageReadMode.RGB)
    src3 = read_image(src3_path, ImageReadMode.RGB)
    
    driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)
    masked_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)

    for i in range(n_frames):
        driver[i] = read_image(driver_paths[i], ImageReadMode.RGB)
        masked_driver[i] = read_image(masked_driver_paths[i], ImageReadMode.RGB)

    data = {
        'n_frames': n_frames,
        'audio': torch.from_numpy(audio).view(1,-1), # (1, L)
        'audio_features': torch.from_numpy(audio_features), # (N, C, L)
        'src1': src1,
        'src2': src2,
        'src3': src3,
        'driver': driver,
        'masked_driver': masked_driver
    }

    return data


def grid_demo2_data_loader(params):
    dataset_root = params['dataset_root']

    # load audio
    audio, _ = librosa.load(join(dataset_root, 'audio.wav'))

    # load features
    audio_features = np.load(join(dataset_root, 'audio_features.npy'))

    # parsing image paths
    src1_path = join(dataset_root, 'src1.jpg')
    src2_path = join(dataset_root, 'src2.jpg')
    src3_path = join(dataset_root, 'src3.jpg')
    
    head_driver_paths = sorted(glob(join(dataset_root, 'head_driver/*.jpg')))
    eye_driver_paths = sorted(glob(join(dataset_root, 'eye_driver/*.jpg')))
    lip_driver_paths = sorted(glob(join(dataset_root, 'lip_driver/*.jpg')))
    masked_driver_paths = sorted(glob(join(dataset_root, 'masked_driver/*.png')))

    # load images
    n_frames = len(audio_features)

    src1 = read_image(src1_path, ImageReadMode.RGB)
    src2 = read_image(src2_path, ImageReadMode.RGB)
    src3 = read_image(src3_path, ImageReadMode.RGB)
    
    head_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)
    eye_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)
    lip_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)
    masked_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)

    for i in range(n_frames):
        head_driver[i] = read_image(head_driver_paths[i], ImageReadMode.RGB)
        eye_driver[i] = read_image(eye_driver_paths[i], ImageReadMode.RGB)
        lip_driver[i] = read_image(lip_driver_paths[i], ImageReadMode.RGB)
        masked_driver[i] = read_image(masked_driver_paths[i], ImageReadMode.RGB)

    data = {
        'n_frames': n_frames,
        'audio': torch.from_numpy(audio).view(1,-1), # (1, L)
        'audio_features': torch.from_numpy(audio_features), # (N, C, L)
        'src1': src1,
        'src2': src2,
        'src3': src3,
        'head_driver': head_driver,
        'eye_driver': eye_driver,
        'lip_driver': lip_driver,
        'masked_driver': masked_driver
    }

    return data


def koeba_demo1_data_loader(params):
    dataset_root = params['dataset_root']

    # load audio
    audio, _ = librosa.load(join(dataset_root, 'audio.wav'))

    # load features
    audio_features = np.load(join(dataset_root, 'audio_features.npy'))

    # parsing image paths
    src1_path = join(dataset_root, 'src1.jpg')
    src2_path = join(dataset_root, 'src2.jpg')
    src3_path = join(dataset_root, 'src3.jpg')
    
    driver_paths = sorted(glob(join(dataset_root, 'driver/*.png')))
    masked_driver_paths = sorted(glob(join(dataset_root, 'masked_driver/*.png')))

    # load images
    n_frames = len(audio_features)

    src1 = read_image(src1_path, ImageReadMode.RGB)
    src2 = read_image(src2_path, ImageReadMode.RGB)
    src3 = read_image(src3_path, ImageReadMode.RGB)
    
    driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)
    masked_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)

    for i in range(n_frames):
        driver[i] = read_image(driver_paths[i], ImageReadMode.RGB)
        masked_driver[i] = read_image(masked_driver_paths[i], ImageReadMode.RGB)

    data = {
        'n_frames': n_frames,
        'audio': torch.from_numpy(audio).view(1,-1), # (1, L)
        'audio_features': torch.from_numpy(audio_features), # (N, C, L)
        'src1': src1,
        'src2': src2,
        'src3': src3,
        'driver': driver,
        'masked_driver': masked_driver
    }

    return data


def koeba_demo2_data_loader(params):
    dataset_root = params['dataset_root']

    # load audio
    audio, _ = librosa.load(join(dataset_root, 'audio.wav'))

    # load features
    audio_features = np.load(join(dataset_root, 'audio_features.npy'))

    # parsing image paths
    src1_path = join(dataset_root, 'src1.jpg')
    src2_path = join(dataset_root, 'src2.jpg')
    src3_path = join(dataset_root, 'src3.jpg')
    
    head_driver_paths = sorted(glob(join(dataset_root, 'head_driver/*.png')))
    eye_driver_paths = sorted(glob(join(dataset_root, 'eye_driver/*.png')))
    lip_driver_paths = sorted(glob(join(dataset_root, 'lip_driver/*.png')))
    masked_driver_paths = sorted(glob(join(dataset_root, 'masked_driver/*.png')))

    # load images
    n_frames = len(audio_features)

    src1 = read_image(src1_path, ImageReadMode.RGB)
    src2 = read_image(src2_path, ImageReadMode.RGB)
    src3 = read_image(src3_path, ImageReadMode.RGB)
    
    head_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)
    eye_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)
    lip_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)
    masked_driver = torch.zeros(n_frames, 3, 256, 256, dtype=torch.uint8)

    for i in range(n_frames):
        head_driver[i] = read_image(head_driver_paths[i], ImageReadMode.RGB)
        eye_driver[i] = read_image(eye_driver_paths[i], ImageReadMode.RGB)
        lip_driver[i] = read_image(lip_driver_paths[i], ImageReadMode.RGB)
        masked_driver[i] = read_image(masked_driver_paths[i], ImageReadMode.RGB)

    data = {
        'n_frames': n_frames,
        'audio': torch.from_numpy(audio).view(1,-1), # (1, L)
        'audio_features': torch.from_numpy(audio_features), # (N, C, L)
        'src1': src1,
        'src2': src2,
        'src3': src3,
        'head_driver': head_driver,
        'eye_driver': eye_driver,
        'lip_driver': lip_driver,
        'masked_driver': masked_driver
    }

    return data
