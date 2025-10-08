import os
import pandas as pd
import numpy as np
import h5py
from scipy.io import wavfile
from scipy.signal import stft
import warnings
from scipy.interpolate import RegularGridInterpolator

# --- Configuration ---
DATA_DIR = 'data'
OUTPUT_DIR = 'processed_data'
SPECTROGRAM_PARAMS = {
    'fs': 44100,  # IMPORTANT: Set this to the sample rate of your WAV files
    'nperseg': 1024,
    'noverlap': 512,
    'min_freq': 300,
    'max_freq': 12000,
    'spec_min_val': 2.0,
    'spec_max_val': 6.5,
    'num_freq_bins': 128,
    'num_time_bins': 128,
}
EPSILON = 1e-12

def create_spectrogram(audio_segment, p):
    """Creates a fixed-size spectrogram from an audio segment (using modern interpolator)."""
    if len(audio_segment) < p['nperseg']:
        return None  # Skip segments that are too short

    # 1. Create STFT
    f, t, spec = stft(audio_segment, fs=p['fs'], nperseg=p['nperseg'], noverlap=p['noverlap'])
    
    # 2. Log transform and normalize
    spec = np.log(np.abs(spec) + EPSILON)
    spec -= p['spec_min_val']
    spec /= (p['spec_max_val'] - p['spec_min_val'])
    spec = np.clip(spec, 0.0, 1.0)

    # 3. Interpolate to a fixed size (128x128) using the modern method
    # The original frequency and time points
    points = (f, t)
    # The new grid we want to interpolate onto
    target_freqs = np.linspace(p['min_freq'], p['max_freq'], p['num_freq_bins'])
    target_times = np.linspace(t.min(), t.max(), p['num_time_bins'])
    grid_y, grid_x = np.meshgrid(target_freqs, target_times, indexing='ij')
    
    # Create the interpolator
    interp_fn = RegularGridInterpolator(points, spec, bounds_error=False, fill_value=0.0)
    
    # Evaluate on the new grid
    fixed_size_spec = interp_fn((grid_y, grid_x))
    
    return fixed_size_spec

def process_file_group(base_filename, audio_file, csv_file, output_path):
    """Processes a single group of audio and csv files."""
    print(f"  Processing: {os.path.basename(csv_file)}...")
    
    try:
        df = pd.read_csv(csv_file, on_bad_lines='skip')
        # Handle potential trailing semicolon
        if df.columns[-1].endswith(';'):
            df = df.rename(columns={df.columns[-1]: df.columns[-1][:-1]})
            df.iloc[:, -1] = df.iloc[:, -1].astype(str).str.rstrip(';')
            
    except Exception as e:
        print(f"    Could not read {csv_file}: {e}")
        return
    
    # Check for required columns
    if 'onset' not in df.columns or 'offset' not in df.columns:
        print(f"    Skipping {csv_file}: missing 'onset' or 'offset' columns.")
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        fs, audio_data = wavfile.read(audio_file)

    if fs != SPECTROGRAM_PARAMS['fs']:
        print(f"    WARNING: Sample rate mismatch in {audio_file}. Expected {SPECTROGRAM_PARAMS['fs']}, got {fs}.")
        return

    spectrograms = []
    for index, row in df.iterrows():
        onset_s = row['onset']
        offset_s = row['offset']
        
        start_sample = int(onset_s * fs)
        end_sample = int(offset_s * fs)
        
        audio_segment = audio_data[start_sample:end_sample]
        
        spec = create_spectrogram(audio_segment, SPECTROGRAM_PARAMS)
        if spec is not None:
            spectrograms.append(spec)

    if spectrograms:
        output_filepath = os.path.join(output_path, f"{base_filename}.hdf5")
        with h5py.File(output_filepath, 'w') as f:
            f.create_dataset('specs', data=np.array(spectrograms, dtype=np.float32))
        print(f"    Saved {len(spectrograms)} spectrograms to {output_filepath}")

def main():
    """Main preprocessing function."""
    print("--- Starting Preprocessing ---")
    
    # Create output directories
    train_output_dir = os.path.join(OUTPUT_DIR, 'train')
    test_output_dir = os.path.join(OUTPUT_DIR, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Find unique audio files
    all_files = os.listdir(DATA_DIR)
    wav_files = {f for f in all_files if f.endswith('.wav')}

    for audio_filename in wav_files:
        base_name = audio_filename.replace('.wav', '')
        print(f"\nProcessing group: {base_name}")
        
        audio_file_path = os.path.join(DATA_DIR, audio_filename)
        
        # Process training data (e.g., ZF.csv)
        train_csv_path = os.path.join(DATA_DIR, f"{base_name}.csv")
        if os.path.exists(train_csv_path):
            process_file_group(base_name + "_train", audio_file_path, train_csv_path, train_output_dir)
            
        # Process testing data (e.g., ZF_test.csv)
        test_csv_path = os.path.join(DATA_DIR, f"{base_name}_test.csv")
        if os.path.exists(test_csv_path):
            process_file_group(base_name + "_test", audio_file_path, test_csv_path, test_output_dir)

    print("\n--- Preprocessing Complete ---")

if __name__ == '__main__':
    main()