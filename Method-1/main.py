import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import os
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler

# Data Augmentation for Features
def augment_features(features, noise_factor=0.01, shift_factor=0.1):
    augmented = features.clone()
    if noise_factor > 0:
        noise = torch.normal(0, noise_factor, size=features.shape, device=features.device)
        augmented += noise
    if shift_factor > 0:
        shift = torch.rand(1, device=features.device) * shift_factor * features.abs().max()
        augmented += shift
    return augmented

# InfoNCEDataset with Augmentation
class InfoNCEDataset(Dataset):
    def __init__(self, csv_path, num_negatives=4, augment=True):
        self.df = pd.read_csv(csv_path)
        self.num_negatives = num_negatives
        self.augment = augment
        self.anchors = []
        self.positives = []
        self.anchor_to_negatives = {}
        self.anchor_to_word = {}
        
        anchor_words = self.df['anchor_word'].unique()
        for word in anchor_words:
            word_df = self.df[self.df['anchor_word'] == word]
            pos_pairs = word_df[word_df['label'] == 1]
            for anchor_path in pos_pairs['anchor_path'].unique():
                anchor_positives = pos_pairs[pos_pairs['anchor_path'] == anchor_path]['pair_path'].tolist()
                self.anchors.append(anchor_path)
                self.positives.append(anchor_positives)
                self.anchor_to_word[anchor_path] = word
                
                neg_pairs = word_df[word_df['label'] == 0]
                self.anchor_to_negatives[anchor_path] = neg_pairs[neg_pairs['anchor_path'] == anchor_path]['pair_path'].tolist()
    
    def __len__(self):
        return len(self.anchors)
    
    def __getitem__(self, idx):
        anchor_path = self.anchors[idx]
        positive_paths = self.positives[idx]
        negative_paths = self.anchor_to_negatives.get(anchor_path, [])
        
        anchor = torch.tensor(np.load(anchor_path), dtype=torch.float32)
        positive_path = random.choice(positive_paths) if positive_paths else anchor_path
        positive = torch.tensor(np.load(positive_path), dtype=torch.float32)
        
        if negative_paths:
            if len(negative_paths) < self.num_negatives:
                sampled_neg_paths = negative_paths + random.choices(negative_paths, k=self.num_negatives - len(negative_paths))
            else:
                sampled_neg_paths = random.sample(negative_paths, self.num_negatives)
            neg_tensors = [torch.tensor(np.load(path), dtype=torch.float32) for path in sampled_neg_paths]
            negatives = torch.stack(neg_tensors)
        else:
            negatives = torch.stack([anchor.clone() for _ in range(self.num_negatives)])
        
        if self.augment:
            anchor = augment_features(anchor)
            positive = augment_features(positive)
            negatives = augment_features(negatives)
        
        return anchor, positive, negatives
        
class ComplexEmbeddingNet(nn.Module):
    def __init__(self, input_dim=1024, embedding_dim=64, dropout=0.4):
        super(ComplexEmbeddingNet, self).__init__()
        self.input_branch1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout)
        )
        self.input_branch2 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.PReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout)
        )
        
        self.block1 = nn.Sequential(
            nn.Linear(2048, 1024), 
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout)
        )
        self.block3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout)
        )
        self.block4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=-1)
        )
        
        self.output_layer = nn.Linear(128, embedding_dim)
        
        self.residual_proj1 = nn.Linear(2048, 1024)
        self.residual_proj2 = nn.Linear(1024, 512)
        self.residual_proj3 = nn.Linear(512, 256)
        self.residual_proj4 = nn.Linear(256, 128)
        self.residual_proj5 = nn.Linear(128, embedding_dim)
        
        self.skip_proj1 = nn.Linear(1024, 512) 
        self.skip_proj2 = nn.Linear(512, 256) 
        self.skip_proj3 = nn.Linear(256, 128)   
    
    def forward(self, x):
        x1 = self.input_branch1(x)  
        x2 = self.input_branch2(x)  
        x = torch.cat([x1, x2], dim=-1) 
        
        x1 = self.block1(x) 
        x = x1 + self.residual_proj1(x)
        skip1 = self.skip_proj1(x) 
        
        x2 = self.block2(x)  
        x = x2 + self.residual_proj2(x) + skip1
        skip2 = self.skip_proj2(x) 
        
        x3 = self.block3(x) 
        x = x3 + self.residual_proj3(x) + skip2
        skip3 = self.skip_proj3(x) 
        
        x4 = self.block4(x)
        x = x4 + self.residual_proj4(x) + skip3
        
        attn_weights = self.attention(x) 
        x = x * attn_weights  
        
        x_out = self.output_layer(x)  
        x = x_out + self.residual_proj5(x)
        
        return x

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, label_smoothing=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
    
    def forward(self, anchor, positive, negatives):
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
        neg_sim = torch.matmul(anchor.unsqueeze(1), negatives.transpose(-2, -1)).squeeze(1) / self.temperature
        
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        return loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = 'best_hindi_audio_embedding_model.pth'
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for anchor, positive, negatives in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negatives = negatives.to(device)  # [batch_size, num_negatives, feature_dim]
            
            optimizer.zero_grad()
            anchor_out = model(anchor)
            positive_out = model(positive)
            
            # Handle negatives
            batch_size, num_negatives, feature_dim = negatives.shape
            negatives_flat = negatives.view(batch_size * num_negatives, feature_dim)
            negatives_out = model(negatives_flat)
            negatives_out = negatives_out.view(batch_size, num_negatives, -1)
            
            loss = criterion(anchor_out, positive_out, negatives_out)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * anchor.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negatives in val_loader:
                anchor = anchor.to(device)
                positive = positive.to(device)
                negatives = negatives.to(device)
                
                anchor_out = model(anchor)
                positive_out = model(positive)
                
                batch_size, num_negatives, feature_dim = negatives.shape
                negatives_flat = negatives.view(batch_size * num_negatives, feature_dim)
                negatives_out = model(negatives_flat)
                negatives_out = negatives_out.view(batch_size, num_negatives, -1)
                
                loss = criterion(anchor_out, positive_out, negatives_out)
                val_loss += loss.item() * anchor.size(0)
        
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
    return model

# Save Validation Data
def save_validation_data(val_dataset, output_dir, dataset):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    
    for idx in tqdm(val_dataset.indices, desc="Saving validation data"):
        anchor, positive, negatives = val_dataset.dataset[idx]
        anchor_path = os.path.join(output_dir, f"anchor_{idx}.npy")
        np.save(anchor_path, anchor.numpy())
        positive_path = os.path.join(output_dir, f"positive_{idx}.npy")
        np.save(positive_path, positive.numpy())
        negative_paths = []
        for j, neg in enumerate(negatives):
            neg_path = os.path.join(output_dir, f"negative_{idx}_{j}.npy")
            np.save(neg_path, neg.numpy())
            negative_paths.append(neg_path)
        
        anchor_orig_path = val_dataset.dataset.anchors[idx]
        anchor_word = dataset.anchor_to_word[anchor_orig_path]
        
        metadata.append({
            'val_idx': idx,
            'anchor_word': anchor_word,
            'anchor_path': anchor_path,
            'positive_path': positive_path,
            'negative_paths': ';'.join(negative_paths)
        })
    
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)  # Fixed path
    print(f"Validation data saved to {output_dir}")

# Verify Saved Validation Data
def verify_validation_data(val_dir):
    metadata = pd.read_csv(os.path.join(val_dir, 'metadata.csv'))
    print(f"Loaded metadata with {len(metadata)} samples")
    
    for _, row in metadata.head(5).iterrows():
        anchor = np.load(row['anchor_path'])
        positive = np.load(row['positive_path'])
        negative_paths = row['negative_paths'].split(';')
        negatives = [np.load(path) for path in negative_paths]
        
        print(f"Sample {row['val_idx']}:")
        print(f"  Word: {row['anchor_word']}")
        print(f"  Anchor shape: {anchor.shape}")
        print(f"  Positive shape: {positive.shape}")
        print(f"  Negatives shapes: {[neg.shape for neg in negatives]}")

# Main Execution
def main():
    # Hyperparameters
    input_dim = 1024
    embedding_dim = 64
    batch_size = 32
    num_epochs = 175
    learning_rate = 0.0001
    temperature = 0.07
    num_negatives = 3
    val_data_dir = 'val_data'
    patience = 5
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = InfoNCEDataset(csv_path='dataset_fixed.csv', num_negatives=num_negatives, augment=True)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    # Save validation data
    save_validation_data(val_dataset, val_data_dir, dataset)
    verify_validation_data(val_data_dir)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss, optimizer, and scheduler
    model = ComplexEmbeddingNet(input_dim, embedding_dim, dropout=0.3).to(device)
    criterion = InfoNCELoss(temperature=temperature, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience)
    
    # Save the final model
    torch.save(model.state_dict(), 'hindi_audio_embedding_infonce_model.pth')

if __name__ == '__main__':
    main()
