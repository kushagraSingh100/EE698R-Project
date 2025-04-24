import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = r"hindi_audio_embedding_infonce_model.pth"
anchor_folder = r"project_root/features/anchors"
test_folder = r"ft"  # New test folder for out-of-sample data

# [Your ComplexEmbeddingNet class remains unchanged]
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

def load_model():
    model = ComplexEmbeddingNet(input_dim=1024, embedding_dim=64, dropout=0.3).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

def get_anchor_embeddings(folder, model):
    embeddings = {}
    words = {}
    
    for filename in os.listdir(folder):
        if filename.endswith('.npy'):
            word = filename.split('_')[0]
            path = os.path.join(folder, filename)
            
            arr = torch.tensor(np.load(path), dtype=torch.float32).to(device)
            with torch.no_grad():
                emb = model(arr.unsqueeze(0)).squeeze(0)
            
            embeddings[filename] = emb.cpu().numpy()
            words[filename] = word
    
    return embeddings, words

def get_test_embeddings(folder, model):
    embeddings = {}
    words = {}
    
    for filename in os.listdir(folder):
        if filename.endswith('.npy'):
            word = filename.split('_')[0]
            path = os.path.join(folder, filename)
            
            arr = torch.tensor(np.load(path), dtype=torch.float32).to(device)
            with torch.no_grad():
                emb = model(arr.unsqueeze(0)).squeeze(0)
            
            embeddings[filename] = emb.cpu().numpy()
            words[filename] = word
    
    return embeddings, words

def calculate_out_sample_mrr(test_embeddings, test_word_map, anchor_embeddings, anchor_word_map):
    test_files = list(test_embeddings.keys())
    anchor_files = list(anchor_embeddings.keys())
    
    test_embs = torch.stack([torch.tensor(emb) for emb in test_embeddings.values()])
    anchor_embs = torch.stack([torch.tensor(emb) for emb in anchor_embeddings.values()])
    
    mrr_scores = []
    
    for i, (test_file, test_emb) in enumerate(test_embeddings.items()):
        test_word = test_word_map[test_file]
        test_word = test_word_map[test_file].split('.')[0]
        test_tensor = torch.tensor(test_emb)
        similarities = cosine_similarity(test_tensor.unsqueeze(0), anchor_embs)
        
        # Get sorted indices (descending order) and convert to list
        sorted_indices = torch.argsort(similarities, descending=True).squeeze(0).tolist()
        
        for rank, idx in enumerate(sorted_indices, 1):
            if anchor_word_map[anchor_files[idx]] == test_word:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)
    
    return np.mean(mrr_scores)

# Main execution
if __name__ == '__main__':
    model = load_model()
    
    # Load anchor embeddings (reference set)
    anchor_embeddings, anchor_word_map = get_anchor_embeddings(anchor_folder, model)
    
    # Load test embeddings (out-of-sample data)
    test_embeddings, test_word_map = get_test_embeddings(test_folder, model)
    
    # Calculate out-of-sample MRR
    mrr = calculate_out_sample_mrr(test_embeddings, test_word_map, anchor_embeddings, anchor_word_map)
    print(f"Out-of-sample MRR: {mrr:.4f}")
