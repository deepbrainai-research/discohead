import torch

from loader import *
from torchvision.io import write_video


def inference(params):
    model = model_loader(params)

    if params['mode'] == 'obama_demo1':
        frames, audio = obama_demo1(model, params)
    elif params['mode'] == 'obama_demo2':
        frames, audio = obama_demo2(model, params)
    elif params['mode'] == 'grid_demo1':
        frames, audio = grid_demo1(model, params)
    elif params['mode'] == 'grid_demo2':
        frames, audio = grid_demo2(model, params)
    elif params['mode'] == 'koeba_demo1':
        frames, audio = koeba_demo1(model, params)
    elif params['mode'] == 'koeba_demo2':
        frames, audio = koeba_demo2(model, params)

    write_video(
        filename=params['save_path'],
        video_array=frames,
        fps=params['fps'],
        video_codec='libx264',
        options={'crf': '12'},
        audio_array=audio,
        audio_fps=params['samplerate'],
        audio_codec='aac')


def obama_demo1(model, params):
    data = obama_demo1_data_loader(params)

    # fixed samples
    src_gpu = data['src'][None].cuda() / 255
    masked_src_gpu = data['masked_src'][None].cuda() / 255
    silence_feature_gpu = data['silence_feature'][None].cuda()
    black = torch.zeros_like(data['src'])

    preds = torch.zeros(data['n_frames'], 3, 256*2, 256*4, dtype=torch.uint8)
    for i in range(data['n_frames']):
        head_driver_gpu = data['head_driver'][i:i+1].cuda() / 255
        masked_driver_gpu = data['masked_driver'][i:i+1].cuda() / 255
        audio_feature_gpu = data['audio_features'][i:i+1].cuda()

        # 1. head only
        ho = model(src_gpu, head_driver_gpu, masked_src_gpu, silence_feature_gpu)
        ho = (ho[0] * 255).type(torch.uint8).cpu()

        # 2. head + audio
        ha = model(src_gpu, head_driver_gpu, masked_src_gpu, audio_feature_gpu)
        ha = (ha[0] * 255).type(torch.uint8).cpu()

        # 3. head + eye + audio
        hea = model(src_gpu, head_driver_gpu, masked_driver_gpu, audio_feature_gpu)
        hea = (hea[0] * 255).type(torch.uint8).cpu()

        preds[i] = torch.cat([
            torch.cat([black, data['head_driver'][i], data['lip_driver'][i], data['eye_driver'][i]], dim=2),
            torch.cat([data['src'], ho, ha, hea], dim=2),
            ], dim=1)

    preds = preds.permute(0, 2, 3, 1)

    return preds, data['audio']


def obama_demo2(model, params):
    data = obama_demo2_data_loader(params)

    # fixed samples
    src_gpu = data['src'][None].cuda() / 255
    masked_src_gpu = data['masked_src'][None].cuda() / 255
    silence_feature_gpu = data['silence_feature'][None].cuda()

    preds = torch.zeros(data['n_frames'], 3, 256, 256*6, dtype=torch.uint8)
    for i in range(data['n_frames']):
        driver_gpu = data['driver'][i:i+1].cuda() / 255
        masked_driver_gpu = data['masked_driver'][i:i+1].cuda() / 255
        audio_feature_gpu = data['audio_features'][i:i+1].cuda()

        # 1. head only
        ho = model(src_gpu, driver_gpu, masked_src_gpu, silence_feature_gpu)
        ho = (ho[0] * 255).type(torch.uint8).cpu()

        # 2. audio only
        ao = model(src_gpu, src_gpu, masked_src_gpu, audio_feature_gpu)
        ao = (ao[0] * 255).type(torch.uint8).cpu()

        # 3. eye only
        eo = model(src_gpu, src_gpu, masked_driver_gpu, silence_feature_gpu)
        eo = (eo[0] * 255).type(torch.uint8).cpu()

        # 4. all
        hea = model(src_gpu, driver_gpu, masked_driver_gpu, audio_feature_gpu)
        hea = (hea[0] * 255).type(torch.uint8).cpu()

        preds[i] = torch.cat([data['src'], data['driver'][i], ho, ao, eo, hea], dim=2)

    preds = preds.permute(0, 2, 3, 1)

    return preds, data['audio']


def grid_demo1(model, params):
    data = grid_demo1_data_loader(params)

    # fixed samples
    src1_gpu = data['src1'][None].cuda() / 255
    src2_gpu = data['src2'][None].cuda() / 255
    src3_gpu = data['src3'][None].cuda() / 255
    black = torch.zeros_like(data['src1'])

    preds = torch.zeros(data['n_frames'], 3, 256*2, 256*4, dtype=torch.uint8)
    for i in range(data['n_frames']):
        driver_gpu = data['driver'][i:i+1].cuda() / 255
        masked_driver_gpu = data['masked_driver'][i:i+1].cuda() / 255
        audio_feature_gpu = data['audio_features'][i:i+1].cuda()

        # 1. head only
        sp1 = model(src1_gpu, driver_gpu, masked_driver_gpu, audio_feature_gpu)
        sp1 = (sp1[0] * 255).type(torch.uint8).cpu()

        # 2. head only + audio
        sp2 = model(src2_gpu, driver_gpu, masked_driver_gpu, audio_feature_gpu)
        sp2 = (sp2[0] * 255).type(torch.uint8).cpu()

        # 3. head only + audio + eye
        sp3 = model(src3_gpu, driver_gpu, masked_driver_gpu, audio_feature_gpu)
        sp3 = (sp3[0] * 255).type(torch.uint8).cpu()

        preds[i] = torch.cat([
            torch.cat([black, data['src1'], data['src2'], data['src3']], dim=2),
            torch.cat([data['driver'][i], sp1, sp2, sp3], dim=2),
            ], dim=1)

    preds = preds.permute(0, 2, 3, 1)

    return preds, data['audio']


def grid_demo2(model, params):
    data = grid_demo2_data_loader(params)

    # fixed samples
    src1_gpu = data['src1'][None].cuda() / 255
    src2_gpu = data['src2'][None].cuda() / 255
    src3_gpu = data['src3'][None].cuda() / 255

    preds = torch.zeros(data['n_frames'], 3, 256*2, 256*4, dtype=torch.uint8)
    for i in range(data['n_frames']):
        head_driver_gpu = data['head_driver'][i:i+1].cuda() / 255
        masked_driver_gpu = data['masked_driver'][i:i+1].cuda() / 255
        audio_feature_gpu = data['audio_features'][i:i+1].cuda()

        # 1. head only
        sp1 = model(src1_gpu, head_driver_gpu, masked_driver_gpu, audio_feature_gpu)
        sp1 = (sp1[0] * 255).type(torch.uint8).cpu()

        # 2. head only + audio
        sp2 = model(src2_gpu, head_driver_gpu, masked_driver_gpu, audio_feature_gpu)
        sp2 = (sp2[0] * 255).type(torch.uint8).cpu()

        # 3. head only + audio + eye
        sp3 = model(src3_gpu, head_driver_gpu, masked_driver_gpu, audio_feature_gpu)
        sp3 = (sp3[0] * 255).type(torch.uint8).cpu()

        half = torch.zeros_like(data['src1'])
        margin = 140
        half[:,:margin] = data['eye_driver'][i][:,:margin]
        half[:,margin:] = data['lip_driver'][i][:,margin:]

        preds[i] = torch.cat([
            torch.cat([data['head_driver'][i], data['src1'], data['src2'], data['src3']], dim=2),
            torch.cat([half, sp1, sp2, sp3], dim=2),
            ], dim=1)

    preds = preds.permute(0, 2, 3, 1)

    return preds, data['audio']


def koeba_demo1(model, params):
    data = koeba_demo1_data_loader(params)

    # fixed samples
    src1_gpu = data['src1'][None].cuda() / 255
    src2_gpu = data['src2'][None].cuda() / 255
    src3_gpu = data['src3'][None].cuda() / 255
    black = torch.zeros_like(data['src1'])

    preds = torch.zeros(data['n_frames'], 3, 256*2, 256*4, dtype=torch.uint8)
    for i in range(data['n_frames']):
        driver_gpu = data['driver'][i:i+1].cuda() / 255
        masked_driver_gpu = data['masked_driver'][i:i+1].cuda() / 255
        audio_feature_gpu = data['audio_features'][i:i+1].cuda()

        # 1. head only
        sp1 = model(src1_gpu, driver_gpu, masked_driver_gpu, audio_feature_gpu)
        sp1 = (sp1[0] * 255).type(torch.uint8).cpu()

        # 2. head only + audio
        sp2 = model(src2_gpu, driver_gpu, masked_driver_gpu, audio_feature_gpu)
        sp2 = (sp2[0] * 255).type(torch.uint8).cpu()

        # 3. head only + audio + eye
        sp3 = model(src3_gpu, driver_gpu, masked_driver_gpu, audio_feature_gpu)
        sp3 = (sp3[0] * 255).type(torch.uint8).cpu()

        preds[i] = torch.cat([
            torch.cat([black, data['src1'], data['src2'], data['src3']], dim=2),
            torch.cat([data['driver'][i], sp1, sp2, sp3], dim=2),
            ], dim=1)

    preds = preds.permute(0, 2, 3, 1)

    return preds, data['audio']


def koeba_demo2(model, params):
    data = koeba_demo2_data_loader(params)

    # fixed samples
    src1_gpu = data['src1'][None].cuda() / 255
    src2_gpu = data['src2'][None].cuda() / 255
    src3_gpu = data['src3'][None].cuda() / 255

    preds = torch.zeros(data['n_frames'], 3, 256*2, 256*4, dtype=torch.uint8)
    for i in range(data['n_frames']):
        head_driver_gpu = data['head_driver'][i:i+1].cuda() / 255
        masked_driver_gpu = data['masked_driver'][i:i+1].cuda() / 255
        audio_feature_gpu = data['audio_features'][i:i+1].cuda()

        # 1. head only
        sp1 = model(src1_gpu, head_driver_gpu, masked_driver_gpu, audio_feature_gpu)
        sp1 = (sp1[0] * 255).type(torch.uint8).cpu()

        # 2. head only + audio
        sp2 = model(src2_gpu, head_driver_gpu, masked_driver_gpu, audio_feature_gpu)
        sp2 = (sp2[0] * 255).type(torch.uint8).cpu()

        # 3. head only + audio + eye
        sp3 = model(src3_gpu, head_driver_gpu, masked_driver_gpu, audio_feature_gpu)
        sp3 = (sp3[0] * 255).type(torch.uint8).cpu()

        half = torch.zeros_like(data['src1'])
        margin = 128
        half[:,:margin] = data['eye_driver'][i][:,:margin]
        half[:,margin:] = data['lip_driver'][i][:,margin:]

        preds[i] = torch.cat([
            torch.cat([data['head_driver'][i], data['src1'], data['src2'], data['src3']], dim=2),
            torch.cat([half, sp1, sp2, sp3], dim=2),
            ], dim=1)

    preds = preds.permute(0, 2, 3, 1)

    return preds, data['audio']
