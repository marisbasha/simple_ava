import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import stft
from scipy.interpolate import RegularGridInterpolator
from torch.distributions import LowRankMultivariateNormal

# --- Configuration ---
# IMPORTANT: These parameters MUST match the ones used in preprocess.py
SPECTROGRAM_PARAMS = {
    'fs': 44100,
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
X_SHAPE = (128, 128)
X_DIM = np.prod(X_SHAPE)

################################################################################
# SECTION 1: VAE MODEL DEFINITION (This is now the FULL model)
################################################################################

class VAE(nn.Module):
    """Exact replica of the VAE from the original AVA repository."""
    def __init__(self, z_dim=32):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self._build_network()

    def _build_network(self):
        # Encoder Layers
        self.conv1 = nn.Conv2d(1, 8, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, 2, padding=1)
        self.conv5 = nn.Conv2d(16, 24, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(24, 24, 3, 2, padding=1)
        self.conv7 = nn.Conv2d(24, 32, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(24)
        self.bn7 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc31 = nn.Linear(256, 64)
        self.fc32 = nn.Linear(256, 64)
        self.fc33 = nn.Linear(256, 64)
        self.fc41 = nn.Linear(64, self.z_dim) # mu
        self.fc42 = nn.Linear(64, self.z_dim) # u
        self.fc43 = nn.Linear(64, self.z_dim) # d

        # Decoder Layers (These must be defined for loading, even if not used)
        self.fc5 = nn.Linear(self.z_dim, 64)
        self.fc6 = nn.Linear(64, 256)
        self.fc7 = nn.Linear(256, 1024)
        self.fc8 = nn.Linear(1024, 8192)
        self.convt1 = nn.ConvTranspose2d(32, 24, 3, 1, padding=1)
        self.convt2 = nn.ConvTranspose2d(24, 24, 3, 2, padding=1, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(24, 16, 3, 1, padding=1)
        self.convt4 = nn.ConvTranspose2d(16, 16, 3, 2, padding=1, output_padding=1)
        self.convt5 = nn.ConvTranspose2d(16, 8, 3, 1, padding=1)
        self.convt6 = nn.ConvTranspose2d(8, 8, 3, 2, padding=1, output_padding=1)
        self.convt7 = nn.ConvTranspose2d(8, 1, 3, 1, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.bn9 = nn.BatchNorm2d(24)
        self.bn10 = nn.BatchNorm2d(24)
        self.bn11 = nn.BatchNorm2d(16)
        self.bn12 = nn.BatchNorm2d(16)
        self.bn13 = nn.BatchNorm2d(8)
        self.bn14 = nn.BatchNorm2d(8)

    def encode(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(self.bn1(x)))
        x = F.relu(self.conv2(self.bn2(x)))
        x = F.relu(self.conv3(self.bn3(x)))
        x = F.relu(self.conv4(self.bn4(x)))
        x = F.relu(self.conv5(self.bn5(x)))
        x = F.relu(self.conv6(self.bn6(x)))
        x = F.relu(self.conv7(self.bn7(x)))
        x = x.view(-1, 8192)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = F.relu(self.fc31(x))
        mu = self.fc41(mu)
        u = F.relu(self.fc32(x))
        u = self.fc42(u).unsqueeze(-1)
        d = F.relu(self.fc33(x))
        d = torch.exp(self.fc43(d))
        return mu, u, d

    def decode(self, z): # We define this but won't call it
        z = F.relu(self.fc5(z))
        z = F.relu(self.fc6(z))
        z = F.relu(self.fc7(z))
        z = F.relu(self.fc8(z))
        z = z.view(-1, 32, 16, 16)
        z = F.relu(self.convt1(self.bn8(z)))
        z = F.relu(self.convt2(self.bn9(z)))
        z = F.relu(self.convt3(self.bn10(z)))
        z = F.relu(self.convt4(self.bn11(z)))
        z = F.relu(self.convt5(self.bn12(z)))
        z = F.relu(self.convt6(self.bn13(z)))
        z = self.convt7(self.bn14(z))
        return z.view(-1, X_DIM)

################################################################################
# SECTION 2: UTILITY FUNCTIONS (Using the modern interpolator to avoid warnings)
################################################################################

def create_spectrogram_from_segment(audio_data, onset_s, offset_s, p):
    """Creates a single, processed spectrogram tensor from an audio segment."""
    fs = p['fs']
    start_sample = int(onset_s * fs)
    end_sample = int(offset_s * fs)
    audio_segment = audio_data[start_sample:end_sample]

    if len(audio_segment) < p['nperseg']:
        raise ValueError("Audio segment is too short for STFT with the given parameters.")

    f, t, spec = stft(audio_segment, fs=fs, nperseg=p['nperseg'], noverlap=p['noverlap'])
    
    spec = np.log(np.abs(spec) + EPSILON)
    spec -= p['spec_min_val']
    spec /= (p['spec_max_val'] - p['spec_min_val'])
    spec = np.clip(spec, 0.0, 1.0)

    points = (f, t)
    target_freqs = np.linspace(p['min_freq'], p['max_freq'], p['num_freq_bins'])
    target_times = np.linspace(t.min(), t.max(), p['num_time_bins'])
    grid_y, grid_x = np.meshgrid(target_freqs, target_times, indexing='ij')
    
    interp_fn = RegularGridInterpolator(points, spec, bounds_error=False, fill_value=0.0)
    fixed_size_spec = interp_fn((grid_y, grid_x))
    
    spec_tensor = torch.from_numpy(fixed_size_spec).type(torch.FloatTensor)
    
    return spec_tensor.unsqueeze(0)

################################################################################
# SECTION 3: MAIN EXECUTION
################################################################################

def main():
    """Main function to extract latent representation."""
    parser = argparse.ArgumentParser(description="Extract the VAE's latent representation for a given audio segment.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.tar file).')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the audio file (.wav).')
    parser.add_argument('--onset', type=float, required=True, help='Onset time of the segment in seconds.')
    parser.add_argument('--offset', type=float, required=True, help='Offset time of the segment in seconds.')
    parser.add_argument('--z_dim', type=int, default=32, help='Latent dimension of the VAE model.')
    args = parser.parse_args()

    # --- 1. Load Model ---
    print(f"Loading model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(z_dim=args.z_dim).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Load Audio ---
    print(f"Loading audio from {args.audio_path}...")
    fs, audio_data = wavfile.read(args.audio_path)
    if fs != SPECTROGRAM_PARAMS['fs']:
        raise ValueError(f"Audio sample rate ({fs}) does not match model's expected rate ({SPECTROGRAM_PARAMS['fs']}).")

    # --- 3. Create Spectrogram ---
    print(f"Generating spectrogram for segment [{args.onset:.3f}s - {args.offset:.3f}s]...")
    try:
        spec_tensor = create_spectrogram_from_segment(audio_data, args.onset, args.offset, SPECTROGRAM_PARAMS)
        spec_tensor = spec_tensor.to(device)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # --- 4. Extract Latent Representation ---
    print("Extracting hidden representation...")
    with torch.no_grad():
        mu, _, _ = model.encode(spec_tensor)
    
    latent_vector = mu.cpu().numpy().flatten()
    
    # --- 5. Print Result ---
    print("\n--- Extraction Complete ---")
    print(f"Latent Representation (mu) of shape {latent_vector.shape}:")
    print(latent_vector)

if __name__ == '__main__':
    main()