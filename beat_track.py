"""
use 3 beat tracker to track the beat of the audio file
save to "\\references" in the dataset folder
"""

import msaf
import os
import librosa
import IPython.display
import core
import jams
import argparse
import madmom

def beat_t_librosa(audio_file):
    sr = core.SR
    hop_length = core.HOP_LEN

    ref_dir = os.path.dirname(os.path.dirname(audio_file)) + '\\references\\'
    audio_name = os.path.basename(audio_file)
    ref_file_path = os.path.join(ref_dir, str(audio_name).replace('.mp3', '.jams'))
    print(ref_file_path)
    # if ref_file_path does not exists
    if not os.path.exists(ref_file_path):
        audio = librosa.load(audio_file, sr=sr)[0]
        _, audio_percussive = librosa.effects.hpss(audio)

        # Compute beats
        tempo, frames = librosa.beat.beat_track(y=audio_percussive, 
                                                sr=sr, hop_length=hop_length)

        # To times
        beat_times = librosa.frames_to_time(frames, sr=sr, 
                                            hop_length=hop_length)
        
        ### write to jams
        jam = jams.JAMS()
        jam.file_metadata.duration = len(audio)/sr

        beat_a = jams.Annotation(namespace='beat')
        beat_a.annotation_metadata = jams.AnnotationMetadata(data_source='librosa beat tracker')

        for t in beat_times:
            beat_a.append(time=t, duration=0.0)

        jam.annotations.append(beat_a)

        ### save to jams
        os.makedirs(ref_dir, exist_ok=True)
        open(ref_file_path, 'w').close()
        jam.save(ref_file_path)

def beat_t_madmom(proc, audio_file):
    '''Korzeniowski beat tracker'''
    sr = core.SR

    ref_dir = os.path.dirname(os.path.dirname(audio_file)) + '\\references\\'
    audio_name = os.path.basename(audio_file)
    ref_file_path = os.path.join(ref_dir, str(audio_name).replace('.mp3', '.jams'))
    print('ref path is:', ref_file_path)
    
    # if ref_file_path does not exists
    if not os.path.exists(ref_file_path):
        audio = librosa.load(audio_file, sr=sr)[0]
        act = madmom.features.beats.RNNBeatProcessor()(audio_file)
        beat_times = proc(act)
 
        ### write to jams
        jam = jams.JAMS()
        jam.file_metadata.duration = len(audio)/sr

        beat_a = jams.Annotation(namespace='beat')
        beat_a.annotation_metadata = jams.AnnotationMetadata(data_source='madmom Korzeniowski beat tracker')

        for t in beat_times:
            beat_a.append(time=t, duration=0.0)

        jam.annotations.append(beat_a)

        ### save to jams
        os.makedirs(ref_dir, exist_ok=True)
        open(ref_file_path, 'w').close()
        jam.save(ref_file_path)

def beat_t_madmom2(proc, audio_file):
    '''Krebs beat tracker'''
    sr = core.SR
    ref_dir = os.path.dirname(os.path.dirname(audio_file)) + '\\references\\'
    audio_name = os.path.basename(audio_file)
    ref_file_path = os.path.join(ref_dir, str(audio_name).replace('.mp3', '.jams'))
    print('ref path is:', ref_file_path)
    
    # if ref_file_path does not exists
    if not os.path.exists(ref_file_path):
        audio = librosa.load(audio_file, sr=sr)[0]
        act = madmom.features.beats.RNNBeatProcessor()(audio_file)
        beat_times = proc(act)
 
        ### write to jams
        jam = jams.JAMS()
        jam.file_metadata.duration = len(audio)/sr

        beat_a = jams.Annotation(namespace='beat')
        beat_a.annotation_metadata = jams.AnnotationMetadata(data_source='madmom Korzeniowski beat tracker')

        for t in beat_times:
            beat_a.append(time=t, duration=0.0)

        jam.annotations.append(beat_a)

        ### save to jams
        os.makedirs(ref_dir, exist_ok=True)
        open(ref_file_path, 'w').close()
        jam.save(ref_file_path)

if __name__ == '__main__':
    # track the beat for all files and save in references folder
    parser = argparse.ArgumentParser(description='beat track function')

    parser.add_argument('--beat_tracker', type=str, choices=['librosa', 'madmom1', 'madmom2'],
                        default='madmom1')
    args = parser.parse_args()

    dataset_dir = core.DATASET_DIR
    audio_dir = os.path.join(dataset_dir, 'audio')

    # define proc
    if args.beat_tracker == 'madmom1':
        proc = madmom.features.beats.CRFBeatDetectionProcessor(fps=100)
    elif args.beat_tracker == 'madmom2':
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)

    for i in range(len(os.listdir(audio_dir))):
        audio_file = os.path.join(audio_dir, os.listdir(audio_dir)[i])
        print(f'processing file: {audio_file}, number is: {i}/{len(os.listdir(audio_dir))}')
        if args.beat_tracker == 'librosa':
            beat_t_librosa(audio_file)
        elif args.beat_tracker == 'madmom1':
            beat_t_madmom(proc, audio_file)
        elif args.beat_tracker == 'madmom2':
            beat_t_madmom2(proc, audio_file)

