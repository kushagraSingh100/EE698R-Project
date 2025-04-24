import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # For progress bar
import random

class ComplexEmbeddingNet(nn.Module):
    def __init__(self, input_dim=1024, embedding_dim=64, dropout=0.4):
        super(ComplexEmbeddingNet, self).__init__()
        # Initial projection with dual branches
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
        
        # Deep residual blocks
        self.block1 = nn.Sequential(
            nn.Linear(2048, 1024),  # Concatenated branches
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
        
        # Attention-based fusion
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=-1)
        )
        
        # Final projection
        self.output_layer = nn.Linear(128, embedding_dim)
        
        # Residual projections
        self.residual_proj1 = nn.Linear(2048, 1024)
        self.residual_proj2 = nn.Linear(1024, 512)
        self.residual_proj3 = nn.Linear(512, 256)
        self.residual_proj4 = nn.Linear(256, 128)
        self.residual_proj5 = nn.Linear(128, embedding_dim)
        
        # Dense skip connections (fixed dimensions)
        self.skip_proj1 = nn.Linear(1024, 512)  # After block1
        self.skip_proj2 = nn.Linear(512, 256)   # After block2
        self.skip_proj3 = nn.Linear(256, 128)   # After block3
    
    def forward(self, x):
        # Dual input branches
        x1 = self.input_branch1(x)  # [batch_size, 1024]
        x2 = self.input_branch2(x)  # [batch_size, 1024]
        x = torch.cat([x1, x2], dim=-1)  # [batch_size, 2048]
        
        # Block 1 with residual and skip
        x1 = self.block1(x)  # [batch_size, 1024]
        x = x1 + self.residual_proj1(x)
        skip1 = self.skip_proj1(x)  # [batch_size, 512]
        
        # Block 2 with residual and skip
        x2 = self.block2(x)  # [batch_size, 512]
        x = x2 + self.residual_proj2(x) + skip1
        skip2 = self.skip_proj2(x)  # [batch_size, 256]
        
        # Block 3 with residual and skip
        x3 = self.block3(x)  # [batch_size, 256]
        x = x3 + self.residual_proj3(x) + skip2
        skip3 = self.skip_proj3(x)  # [batch_size, 128]
        
        # Block 4 with residual
        x4 = self.block4(x)  # [batch_size, 128]
        x = x4 + self.residual_proj4(x) + skip3
        
        # Attention-based fusion
        attn_weights = self.attention(x)  # [batch_size, 1]
        x = x * attn_weights  # [batch_size, 128]
        
        # Output with residual
        x_out = self.output_layer(x)  # [batch_size, embedding_dim]
        x = x_out + self.residual_proj5(x)
        
        return x
        

# Simplified Dataset for Testing (only anchors and their words)
class InfoNCEDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.anchors = []
        self.anchor_to_word = {}
        
        # Group by anchor word
        anchor_words = self.df['anchor_word'].unique()
        for word in anchor_words:
            word_df = self.df[self.df['anchor_word'] == word]
            pos_pairs = word_df[word_df['label'] == 1]
            for anchor_path in pos_pairs['anchor_path'].unique():
                self.anchors.append(anchor_path)
                self.anchor_to_word[anchor_path] = word
    
    def __len__(self):
        return len(self.anchors)
    
    def __getitem__(self, idx):
        anchor_path = self.anchors[idx]
        anchor = torch.tensor(np.load(anchor_path), dtype=torch.float32)
        return anchor, anchor_path

# Function to find the top 10 closest anchor words
def find_top_10_words(test_feature_path, model, dataset, device, batch_size=32):
    """
    Find the top 10 closest anchor words to a test audio sample using cosine similarity.
    
    Args:
        test_feature_path (str): Path to the test audio's .npy feature file.
        model (nn.Module): Trained EmbeddingNet model.
        dataset (InfoNCEDataset): Dataset with anchor paths and words.
        device (torch.device): Device to run computations (cuda or cpu).
        batch_size (int): Batch size for processing anchor embeddings.
    
    Returns:
        list of tuples: [(word, similarity_score, anchor_path), ...] for top 10 matches.
    """
    model.eval()
    
    # Load and compute embedding for test sample
    test_feature = torch.tensor(np.load(test_feature_path), dtype=torch.float32).to(device)
    with torch.no_grad():
        test_embedding = model(test_feature.unsqueeze(0))  # [1, embedding_dim]
        test_embedding = F.normalize(test_embedding, dim=-1)  # Normalize for cosine similarity
    
    # Compute embeddings for all anchors using batch processing
    anchor_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    anchor_embeddings = []
    anchor_paths = []
    
    for anchors, paths in tqdm(anchor_loader, desc="Computing anchor embeddings"):
        anchors = anchors.to(device)
        with torch.no_grad():
            embeddings = model(anchors)  # [batch_size, embedding_dim]
            embeddings = F.normalize(embeddings, dim=-1)
        anchor_embeddings.append(embeddings)
        anchor_paths.extend(paths)
    
    anchor_embeddings = torch.cat(anchor_embeddings, dim=0)  # [num_anchors, embedding_dim]
    
    # Compute cosine similarities
    similarities = torch.matmul(test_embedding, anchor_embeddings.T).squeeze(0)  # [num_anchors]
    
    # Get top 10 matches
    top_k = 25
    top_k_scores, top_k_indices = torch.topk(similarities, k=top_k, dim=0)
    top_k_scores = top_k_scores.cpu().numpy()
    top_k_indices = top_k_indices.cpu().numpy()
    
    # Map to words and paths
    results = []
    for idx, score in zip(top_k_indices, top_k_scores):
        anchor_path = anchor_paths[idx]
        word = dataset.anchor_to_word[anchor_path]
        results.append((word, score, anchor_path))
    
    return results

# Main Testing Function
def main():
    # Configuration
    model_path = 'hindi_audio_embedding_infonce_model.pth'
    csv_path = 'dataset_fixed.csv'
    test_feature_path = 'ft/स्वास्थ्य.npy'
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = ComplexEmbeddingNet(input_dim=1024, embedding_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Initialize dataset
    dataset = InfoNCEDataset(csv_path=csv_path)
    
    # Find top 10 closest words
    results = find_top_10_words(test_feature_path, model, dataset, device, batch_size=batch_size)
    
    # Print results
    print(f"\nTest audio: {test_feature_path}")
    for i, (word, score, anchor_path) in enumerate(results, 1):
        print(f"Rank {i}: Word = {word}, Similarity = {score:.4f}, Anchor Path = {anchor_path}")

if __name__ == '__main__':
    main()
