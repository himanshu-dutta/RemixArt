import os
import time
import pandas as pd
import argparse
from utils import get_note_string

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Text Embedding Training')

    parser.add_argument('--midi_dir', type=str)
    parser.add_argument('--save_path', type=str)
    args = vars(parser.parse_args())

    midi_dir = args['midi_dir']
    save_path = args['save_path']
    save_path = os.path.join(save_path, 'audio_vector.csv')
    files = [os.path.join(midi_dir, f) for f in os.listdir(midi_dir)]
    midi_vector = []
    start_time = time.time()
    i = 0
    for f in files:
        ns = get_note_string(f)
        if ns:
            midi_vector.append([os.path.basename(f).split('.')[0], ns])

        i += 1
        if i % 100 == 0:
            ns_df = pd.DataFrame(midi_vector, columns=[
                                 'id', 'notes']).set_index('id')
            ns_df.to_csv(save_path)
            print(f'{i}/{len(files)} processed...')
    print('Finished in {} mins'.format((time.time() - start_time)/60))
