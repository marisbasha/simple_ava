# train.py
import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.distributions import LowRankMultivariateNormal

# Constants from the original AVA implementation
X_SHAPE = (128, 128)
X_DIM = np.prod(X_SHAPE)

################################################################################
# SECTION 1: PYTORCH DATASET AND DATALOADER (No changes needed)
################################################################################

def numpy_to_tensor(x):
    return torch.from_numpy(x).type(torch.FloatTensor)

class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.filenames = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith('.hdf5')]
        
        self.file_lengths = []
        self.total_specs = 0
        for f in self.filenames:
            with h5py.File(f, 'r') as hf:
                length = len(hf['specs'])
                self.file_lengths.append(length)
                self.total_specs += length
        
        self.cumulative_lengths = np.cumsum(self.file_lengths)

    def __len__(self):
        return self.total_specs

    def __getitem__(self, index):
        file_idx = np.searchsorted(self.cumulative_lengths, index, side='right')
        local_idx = index
        if file_idx > 0:
            local_idx = index - self.cumulative_lengths[file_idx - 1]
            
        with h5py.File(self.filenames[file_idx], 'r') as hf:
            spec = hf['specs'][local_idx]
            
        if self.transform:
            spec = self.transform(spec)
            
        return spec

def get_data_loaders(base_dir, batch_size=64, num_workers=4):
    loaders = {}
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    
    if os.path.exists(train_dir) and os.listdir(train_dir):
        train_dataset = SpectrogramDataset(train_dir, transform=numpy_to_tensor)
        loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    if os.path.exists(test_dir) and os.listdir(test_dir):
        test_dataset = SpectrogramDataset(test_dir, transform=numpy_to_tensor)
        loaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    return loaders

################################################################################
# SECTION 2: VAE MODEL (Exact replica of the original AVA network)
################################################################################

class VAE(nn.Module):
    """Exact replica of the VAE from the original AVA repository."""
    def __init__(self, save_dir='', lr=1e-3, z_dim=32, model_precision=10.0):
        super(VAE, self).__init__()
        self.save_dir = save_dir
        self.lr = lr
        self.z_dim = z_dim
        self.model_precision = model_precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._build_network()
        
        self.optimizer = Adam(self.parameters(), lr=self.lr)
        self.epoch = 0
        os.makedirs(save_dir, exist_ok=True)
        self.to(self.device)

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
        self.fc42 = nn.Linear(64, self.z_dim) # u (low-rank factor)
        self.fc43 = nn.Linear(64, self.z_dim) # d (diagonal factor)

        # Decoder Layers
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

    def decode(self, z):
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

    def forward(self, x):
        mu, u, d = self.encode(x)
        latent_dist = LowRankMultivariateNormal(mu, u, d)
        z = latent_dist.rsample()
        x_rec = self.decode(z)
        
        # ELBO Loss Calculation (as in original paper)
        # 1. Prior Term: E_{q(z|x)}[log p(z)]
        log_pz = -0.5 * (torch.sum(torch.pow(z, 2)) + self.z_dim * np.log(2 * np.pi))
        
        # 2. Reconstruction Term: E_{q(z|x)}[log p(x|z)]
        log_pxz_term = -0.5 * X_DIM * (np.log(2 * np.pi / self.model_precision))
        l2s = torch.sum(torch.pow(x.view(x.shape[0], -1) - x_rec, 2), dim=1)
        log_pxz = log_pxz_term - 0.5 * self.model_precision * torch.sum(l2s)
        
        # 3. Entropy Term: H[q(z|x)]
        entropy_q = torch.sum(latent_dist.entropy())
        
        elbo = log_pz + log_pxz + entropy_q
        
        # The loss is the negative ELBO
        return -elbo

    def train_loop(self, loaders, epochs, test_freq=5, save_freq=10):
        print("="*40)
        print("Training Original AVA VAE...")
        print(f"Device: {self.device}")
        
        for epoch in range(1, epochs + 1):
            self.train()
            train_loss = 0
            for data in loaders['train']:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                loss = self.forward(data) # Forward pass now returns the loss directly
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
            
            avg_train_loss = train_loss / len(loaders['train'].dataset)
            print(f'====> Epoch: {epoch} Average Train Loss: {avg_train_loss:.4f}')

            if 'test' in loaders and epoch % test_freq == 0:
                self.test(loaders['test'], epoch)

            if epoch % save_freq == 0:
                self.save_state(f"checkpoint_{str(epoch).zfill(3)}.tar")

    def test(self, test_loader, epoch):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                loss = self.forward(data)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader.dataset)
        print(f'====> Epoch {epoch} Test Loss: {avg_test_loss:.4f}')

    def save_state(self, filename):
        state = {'state_dict': self.state_dict(), 'optimizer': self.optimizer.state_dict()}
        filepath = os.path.join(self.save_dir, filename)
        torch.save(state, filepath)
        print(f"Model saved to {filepath}")

################################################################################
# SECTION 3: MAIN EXECUTION
################################################################################

if __name__ == '__main__':
    # --- Configuration ---
    PROCESSED_DATA_DIR = 'processed_data'
    MODEL_SAVE_DIR = 'model_checkpoints'
    
    LEARNING_RATE = 1e-3
    LATENT_DIM = 32
    MODEL_PRECISION = 10.0 # Controls reconstruction/regularization trade-off
    EPOCHS = 100
    BATCH_SIZE = 64
    TEST_FREQUENCY = 5
    SAVE_FREQUENCY = 20

    # --- Step 1: Prepare DataLoaders ---
    print("\n--- Preparing DataLoaders ---")
    loaders = get_data_loaders(PROCESSED_DATA_DIR, batch_size=BATCH_SIZE)
    if not loaders:
        print("No data found. Did you run preprocess.py first?")
        exit()
    print(f"Found {len(loaders.get('train', []).dataset)} training samples.")
    if 'test' in loaders:
        print(f"Found {len(loaders['test'].dataset)} testing samples.")
    print("--- DataLoaders Ready ---")

    # --- Step 2: Train the VAE model ---
    model = VAE(
        save_dir=MODEL_SAVE_DIR, 
        lr=LEARNING_RATE, 
        z_dim=LATENT_DIM,
        model_precision=MODEL_PRECISION
    )
    model.train_loop(loaders, epochs=EPOCHS, test_freq=TEST_FREQUENCY, save_freq=SAVE_FREQUENCY)
    
    print("\n--- VAE Training Complete ---")