import numpy as np
import pretty_midi



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
        
