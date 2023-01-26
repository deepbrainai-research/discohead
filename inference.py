import torch

from loader import *
from torchvision.io import write_video


def inference(params):
    model = model_loader(params)

    if params['fig_number'] == '2':
        frames, audio = fig2(model, params)
    elif params['fig_number'] == '3':
        frames, audio = fig3(model, params)
    elif params['fig_number'] == '4':
        frames, audio = fig4(model, params)

    write_video(
        filename=params['save_path'],
        video_array=frames,
        fps=params['fps'],
        video_codec='libx264',
        options={'crf': '12'},
        audio_array=audio,
        audio_fps=params['samplerate'],
        audio_codec='aac')


def fig2(model, params):
    data = fig2_data_loader(params)

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



def fig3(model, params):
    data = fig3_data_loader(params)

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


def fig4(model, params):
    data = fig4_data_loader(params)

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
