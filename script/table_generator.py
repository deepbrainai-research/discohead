import os 

wavfiles = sorted(os.listdir('assets/audio/_gt'))
models = sorted(os.listdir('assets/audio'))

for category in ['gt', 'tt']:
    for wavfile in wavfiles:
        row = []
        for model in models:
            if model == '_gt':
                row.append(f'<audio src="https://raw.githubusercontent.com/moneybrain-research/gan-vocoder/master/assets/audio/_gt/{wavfile}" controls preload="auto">')
            else:
                row.append(f'<audio src="https://raw.githubusercontent.com/moneybrain-research/gan-vocoder/master/assets/audio/{model}/{category}/{wavfile}" controls preload="auto">')
                # row.append(f'<audio src="assets/audio/{model}/{category}/{wavfile}" controls preload="auto">')

        row = '|' + '|'.join(row) + '|'
        print(row)
    print('-'*50)

