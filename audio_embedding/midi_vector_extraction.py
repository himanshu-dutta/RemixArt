import os, time
import numpy as np
import pretty_midi
import pandas as pd
import argparse


def parse_midi(path):
    midi = None
    with open(path, 'rb') as f:
        try:
            midi = pretty_midi.PrettyMIDI(f)
            midi.remove_invalid_notes()
        except:
            return None
    return midi

def get_percent_monophonic(pm_instrument_roll):
    mask = pm_instrument_roll.T > 0
    notes = np.sum(mask, axis=1)
    n = np.count_nonzero(notes)
    single = np.count_nonzero(notes == 1)
    if single > 0:
        return float(single) / float(n)
    elif single == 0 and n > 0:
        return 0.0
    else: # no notes of any kind
        return 0.0
    
def filter_monophonic(pm_instruments, percent_monophonic=0.99):
    return [i for i in pm_instruments if get_percent_monophonic(i.get_piano_roll()) >= percent_monophonic]

def sort_by_start(note):
    return float(note.start)

def get_note_string(midi):
    midi = parse_midi(f)
    if midi is not None:
        for instrument in midi.instruments:
            buff = [n for n in instrument.notes]
        buff.sort(key=sort_by_start)
        buff = [str(n.pitch) for n in buff]
        return buff
    else: 
        return None
        
        
        
if __name__=='__main__':
       
    parser = argparse.ArgumentParser('Text Embedding Training')

    parser.add_argument('--midi_dir',type=str)
    parser.add_argument('--save_path',type=str)
    args = vars(parser.parse_args())
    
    midi_dir = args['midi_dir']
    save_path = args['save_path']
    save_path = os.path.join(save_path,'audio_vector.csv')
    files = [os.path.join(midi_dir, f) for f in os.listdir(midi_dir)]
    midi_vector = []
    start_time = time.time()
    i=0
    for f in files:
        ns = get_note_string(f)
        if ns:
            midi_vector.append([os.path.basename(files[0]).split('.')[0],ns])
            
        i+=1
        if i%100==0:
            ns_df = pd.DataFrame(midi_vector,columns=['id','notes']).set_index('id')
            ns_df.to_csv(save_path)
            print(f'{i}/{len(files)} processed...')
    print('Finished in {} mins'.format((time.time() - start_time)/60))

