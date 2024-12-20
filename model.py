import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import ViTEmbedder
from hubert import AudioEmbedder
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
class AudioVisualModel(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        
        self.visual_embedder = ViTEmbedder()
        self.audio_embedder = AudioEmbedder()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Unfreeze the HuBERT model
        for param in self.audio_embedder.hubert.parameters():
            param.requires_grad = True
        
        for param in self.audio_embedder.projection.parameters():
            param.requires_grad = True
        
    def compute_similarity_matrix(self, audio_feats, visual_feats): #ye take this
        """
        Compute pairwise cosine similarities between audio and visual tokens
        
        Args:
            audio_feats: (B, Na, D)  # B=batch, Na=num_audio_tokens, D=embedding_dim
            visual_feats: (B, Nv, D) # Nv=num_visual_tokens
            
        Returns:
            similarity_matrix: (B, Na, Nv)
        """
        # Normalize embeddings
        #print("Audio feats stats before norm - min:", audio_feats.min().item(), "max:", audio_feats.max().item())
        #print("Visual feats stats before norm - min:", visual_feats.min().item(), "max:", visual_feats.max().item())
        
        # Normalize embeddings
        audio_feats = F.normalize(audio_feats, dim=-1)  #this has to be checked if we wanna fucking normalize the embeddings
        visual_feats = F.normalize(visual_feats, dim=-1) #same
        
        # Compute similarities and check values
        similarity = torch.bmm(audio_feats, visual_feats.transpose(1, 2))
        #print("Raw similarity stats - min:", similarity.min().item(),
          #  "max:", similarity.max().item())
        
        return similarity / self.temperature
    
    def aggregate_token_similarities(self, similarity_matrix): #also take this
        """
        Aggregate token-level similarities using max-mean strategy
        
        Args:
            similarity_matrix: (B, Na, Nv)
            
        Returns:
            clip_similarity: (B)
        """
        # Max pool over visual dimension for each audio token
        max_similarities = torch.max(similarity_matrix, dim=2)[0]  # (B, Na)
        
        # Average over audio tokens
        clip_similarity = torch.mean(max_similarities, dim=1)  # (B)
        return clip_similarity
    
    def compute_all_similarities(self, audio_feats, visual_feats):
        """Compute similarities between all pairs of audio and visual features in batch"""
        B = audio_feats.shape[0]
        
        audio_feats = audio_feats.unsqueeze(1).expand(-1, B, -1, -1)
        visual_feats = visual_feats.unsqueeze(0).expand(B, -1, -1, -1)
        #print("During compute_all_similarities")
        #print(audio_feats.shape, visual_feats.shape)
        # Normalize features
        audio_feats = F.normalize(audio_feats, dim=-1)
        visual_feats = F.normalize(visual_feats, dim=-1)
        
        # Compute token-level similarities
        token_sims = torch.matmul(
            audio_feats, 
            visual_feats.transpose(2, 3)
        ) / self.temperature
        
        # Aggregate using max-mean strategy
        max_sims = torch.max(token_sims, dim=3)[0]  # Max over visual dimension (B, B, Na)
        clip_sims = torch.mean(max_sims, dim=2)     # Mean over audio dimension (B, B)
        
        return clip_sims, token_sims

    def compute_contrastive_loss(self, clip_similarities, token_sims):
        """Compute InfoNCE loss with regularization"""
        batch_size = clip_similarities.shape[0]
        labels = torch.arange(batch_size).to(clip_similarities.device)
        
        # Audio to Visual direction
        log_prob_a2v = F.log_softmax(clip_similarities, dim=1)
        losses_a2v = -log_prob_a2v[torch.arange(batch_size), labels]
        
        # Visual to Audio direction  
        log_prob_v2a = F.log_softmax(clip_similarities.t(), dim=1)
        losses_v2a = -log_prob_v2a[torch.arange(batch_size), labels]
        
        # Average both directions
        contrastive_loss = (losses_a2v + losses_v2a).mean() / 2
        
        # Add regularization
        reg_loss = self.compute_regularization_losses(clip_similarities, token_sims)
        
        total_loss = contrastive_loss + reg_loss
        #print("Regularization loss", reg_loss)
        #print("Contrastive loss", contrastive_loss)
        return total_loss
    

    def compute_regularization_losses(self, clip_sims, token_sims):
            # 1. Non-negative pressure (unchanged)
            neg_sims = torch.clamp(token_sims, min=-20, max=0)  
            l_nonneg = torch.mean(neg_sims ** 2)
            
            # 2. Temperature regularization (fixed to handle both bounds)
            temp_low = torch.clamp(torch.log(torch.tensor(2.3, device=token_sims.device)) - torch.log(self.temperature), min=0) ** 4
            temp_high = torch.clamp(torch.log(self.temperature) - torch.log(torch.tensor(4.0, device=token_sims.device)), min=0) ** 4
            l_cal = temp_low + temp_high
            
            # 3. Spatial smoothness only (removed incorrect temporal)
            spatial_diffs = token_sims[..., 1:] - token_sims[..., :-1]
            l_spatial = torch.mean(spatial_diffs ** 2)
            
            # 4. Sparsity with normalized entropy
            attn_norm = torch.sigmoid(token_sims)
            threshold = 0.5
            above_threshold = F.relu(attn_norm - threshold)
            num_high_attn = torch.sum(above_threshold, dim=-1)
            
            attn_dist = F.softmax(token_sims, dim=-1)
            entropy = -(attn_dist * torch.log(attn_dist + 1e-10)).sum(dim=-1)
            # Normalize entropy by maximum possible entropy (log of number of elements)
            max_entropy = torch.log(torch.tensor(attn_dist.size(-1), dtype=torch.float, device=token_sims.device))
            normalized_entropy = entropy / max_entropy
            
            l_sparsity = torch.mean(num_high_attn ** 2) - 0.1 * torch.mean(normalized_entropy)
            
            reg_loss = ( 8.0 * l_cal )                 #                 0.15 * l_nonneg + 
                         #+ 
                        #0.01 * l_spatial +
                        #0.005 * l_sparsity)
            
            return reg_loss
        
    def forward(self, frames, audio):
        """
        Forward pass computing embeddings, similarities and loss
        
        Args:
            frames: (B, C, H, W) batch of video frames
            spectrograms: (B, T, F) batch of audio spectrograms
            
        Returns:
            loss if training, clip_similarities if not
        """
        # Get embeddings
        visual_feats = self.visual_embedder(frames)
        audio_feats = self.audio_embedder(audio)
        
        if self.training:
            # Get similarities and token-level similarities
            clip_sims, token_sims = self.compute_all_similarities(audio_feats, visual_feats)
            return self.compute_contrastive_loss(clip_sims, token_sims)
        else:
            # During inference, just get clip similarities
            token_sims = self.compute_similarity_matrix(audio_feats, visual_feats)
            return token_sims

if __name__ == "__main__":
    # Test the model
    model = AudioVisualModel()
    
    # Create dummy batch
    batch_size = 4
    frames = torch.randn(batch_size, 3, 224, 224)
    audio = torch.randn(batch_size, 16331)
    
    # Test training mode
    loss = model(frames, audio)
    print(f"Training loss: {loss.item()}")
    
    # Test inference mode
    model.eval()
    
    similarities = model(frames, audio)
    print(f"Inference similarities shape: {similarities.shape}")  # Should be (batch_size)
    print(f"Similarity values: {similarities}")
