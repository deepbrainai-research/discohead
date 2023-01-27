if __name__ == "__main__":
    import argparse

    from inference import inference

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', required=True)
    args = parser.parse_args()

    mode = args.mode

    # parsing mode
    if mode == 'obama_demo1':
        from config_obama import params
        params['mode'] = mode
        params['dataset_root'] = './dataset/obama/demo1'
        params['save_path'] = 'obama_demo1.mp4'
    elif mode == 'obama_demo2':
        from config_obama import params
        params['mode'] = mode
        params['dataset_root'] = './dataset/obama/demo2'
        params['save_path'] = 'obama_demo2.mp4'
    elif mode == 'grid_demo1':
        from config_grid import params
        params['mode'] = mode
        params['dataset_root'] = './dataset/grid/demo1'
        params['save_path'] = 'grid_demo1.mp4'
    elif mode == 'grid_demo2':
        from config_grid import params
        params['mode'] = mode
        params['dataset_root'] = './dataset/grid/demo2'
        params['save_path'] = 'grid_demo2.mp4'
    elif mode == 'koeba_demo1':
        from config_koeba import params
        params['mode'] = mode
        params['dataset_root'] = './dataset/koeba/demo1'
        params['save_path'] = 'koeba_demo1.mp4'
    elif mode == 'koeba_demo2':
        from config_koeba import params
        params['mode'] = mode
        params['dataset_root'] = './dataset/koeba/demo2'
        params['save_path'] = 'koeba_demo2.mp4'
    else:
        raise Exception('mode: [obama_demo1|obama_demo2|grid_demo1|grid_demo2|koeba_demo1|koeba_demo2]')

    inference(params)
