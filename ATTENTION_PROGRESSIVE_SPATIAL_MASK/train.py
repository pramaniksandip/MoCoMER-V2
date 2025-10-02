import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import DataPreperation
from DataModule import DataModule, collate_fn
from DataModule import ScaleAugmentation, ScaleToLimitRange
from DenseNet import DenseNet
from Self_Attn import SelfAttention
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

# Define the surrogate attention network
class SurrogateAttentionNet(nn.Module):
    def __init__(self, pretrained_model_path=None, device='cuda'):
        super(SurrogateAttentionNet, self).__init__()
        self.device = device
        self.feature_extractor = DenseNet(growth_rate=32, num_layers=4, reduction=0.5)
        
        # Load pre-trained model if provided
        if pretrained_model_path:
            state_dict = torch.load(pretrained_model_path, map_location=device)
            self.feature_extractor.load_state_dict(state_dict, strict=False)
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.attention = SelfAttention(self.feature_extractor.out_channels)

    def forward(self, x, mask):
        feature_map, _ = self.feature_extractor(x.to(self.device), None)
        
        # Ensure mask has the same spatial dimensions as feature_map
        mask = F.interpolate(mask, size=(feature_map.shape[2], feature_map.shape[3]), mode='nearest')
        masked_feature_map = feature_map * mask  # Apply progressive mask
        
        att_map, _ = self.attention(masked_feature_map)
        
        return att_map

# Function to generate progressive mask
def generate_progressive_mask(batch_size, height, width, epoch, total_epochs, device='cuda'):
    min_reveal = 0.3  # Start with at least 30% visibility
    max_reveal = 1.0  # Fully revealed at the last epoch
    reveal_ratio = min_reveal + (epoch / total_epochs) * (max_reveal - min_reveal)
    
    mask = torch.rand(batch_size, 1, height, width, device=device) < reveal_ratio
    return mask.float()

# Modified Training Loop with TQDM and enhanced printing
def train_model(model, dataloader, optimizer, num_epochs=25):
    """
    Trains the model with a progress bar for each epoch and improved logging.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🚀 Starting training on {device} for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        samples_processed = 0
        
        # Wrap the dataloader with tqdm for a progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True, unit="batch")
        
        for inputs, x_mask, _ in progress_bar:
            inputs = inputs.to(device)
            
            # Generate progressive mask based on current epoch
            batch_size, _, height, width = inputs.size()
            mask = generate_progressive_mask(batch_size, height, width, epoch, num_epochs, device=device)
            
            # Forward pass
            optimizer.zero_grad()
            surrogate_attention_map = model(inputs, mask)
            
            # Compute attention supervision loss
            model_attention_map, _ = model.attention(model.feature_extractor(inputs)[0])
            att_loss = torch.nn.functional.mse_loss(surrogate_attention_map, model_attention_map)
            
            # Backward pass and optimization
            att_loss.backward()
            optimizer.step()
            
            # Update running loss and the progress bar's postfix
            running_loss += att_loss.item() * inputs.size(0)
            samples_processed += inputs.size(0)
            current_avg_loss = running_loss / samples_processed
            progress_bar.set_postfix(loss=f"{current_avg_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)
            # Use tqdm.write to print messages without disrupting the progress bar
            tqdm.write(f"✅ Checkpoint saved at Epoch {epoch + 1}: {save_path}")

    print("\n🎉 Training finished!")
    final_model_path = "surrogate_attention_net_hme100k.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Final model saved: {final_model_path}")


# --- The rest of your script remains unchanged ---

# Data Preparation
root_path = "/workspace/nemo/data/RESEARCH/HMER_MASK_MATCH_HME100K/HME100K/train_images/"  
data = DataPreperation(root_path=root_path)
train_df = data.Train_Test_Data()

# Define constants
K_MIN = 0.7
K_MAX = 1.4
H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024

cv2_transforms = transforms.Compose([
    ScaleAugmentation(K_MIN, K_MAX),
    ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI)
])

pil_transforms = transforms.Compose([
    transforms.ToTensor(),
])

dm = DataModule(bs=16, root_path=root_path, df=train_df, cv2_transforms=cv2_transforms, pil_transforms=pil_transforms)
dataloader = DataLoader(dm, batch_size=16, collate_fn=collate_fn, shuffle=True)

# Initialize model and optimizer
model = SurrogateAttentionNet(device='cuda')
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

# Train the model
train_model(model, dataloader, optimizer, num_epochs=300)