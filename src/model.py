import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import HubertModel, AutoProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class ViTEmbedder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') #torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.projection = nn.Linear(model.embed_dim, 512)
        
        for param in self.model.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            patch_embeddings: (batch_size, num_patches, embedding_dim)
        """
        x = self.model.get_intermediate_layers(x, n=1)[0]
        x = self.projection(x)
        return x

class AudioEmbedder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960") #model name: facebook/hubert-large-ls960-ft
        self.projection = nn.Linear(self.hubert.config.hidden_size, embedding_dim)  
        for param in self.hubert.parameters():
            param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True
        
    def forward(self, audio_input):
        """
        Args:
            audio_input: (B, T) raw audio waveform at 16kHz
            
        Returns:
            features: (B, Na, D) where:
                B is batch size
                Na is number of audio tokens
                D is embedding_dim
        """
        inputs = self.processor(
            audio_input, 
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True
        ).input_values.squeeze(0)
        inputs = inputs.to(audio_input.device)
        hubert_output = self.hubert(inputs).last_hidden_state  # (B, T/320, 1024)
        features = self.projection(hubert_output)  # (B, T/320, embedding_dim)
        
        return features

class AudioVisualModel(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        
        self.visual_embedder = ViTEmbedder()
        self.audio_embedder = AudioEmbedder()
        self.temperature = nn.Parameter(torch.tensor(temperature))

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
        #audio_feats = F.normalize(audio_feats, dim=-1)  
        #visual_feats = F.normalize(visual_feats, dim=-1)
        similarity = torch.bmm(audio_feats, visual_feats.transpose(1, 2))
        return similarity / self.temperature
    
    def aggregate_token_similarities(self, similarity_matrix):
        """
        Aggregate token-level similarities using max-mean strategy
        
        Args:
            similarity_matrix: (B, Na, Nv)
            
        Returns:
            clip_similarity: (B)
        """
        max_similarities = torch.max(similarity_matrix, dim=2)[0]  # (B, Na)
        clip_similarity = torch.mean(max_similarities, dim=1)  # (B)
        return clip_similarity
    
    def compute_all_similarities(self, audio_feats, visual_feats):
        """Compute similarities between all pairs of audio and visual features in batch"""
        B = audio_feats.shape[0]
        
        audio_feats = audio_feats.unsqueeze(1).expand(-1, B, -1, -1)
        visual_feats = visual_feats.unsqueeze(0).expand(B, -1, -1, -1)
        #audio_feats = F.normalize(audio_feats, dim=-1)
        #visual_feats = F.normalize(visual_feats, dim=-1)
        
        # token-level similarities
        token_sims = torch.matmul(
            audio_feats, 
            visual_feats.transpose(2, 3)
        ) / self.temperature
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
        reg_loss = self.compute_regularization_losses(clip_similarities, token_sims)    
        total_loss = contrastive_loss + reg_loss
        return total_loss
    

    def compute_regularization_losses(self, clip_sims, token_sims):
            # 1. Non-negative pressure (unchanged)
            neg_sims = torch.clamp(token_sims, min=-20, max=0)  
            l_nonneg = torch.mean(neg_sims ** 2)
            
            # 2. Temperature regularization (fixed to handle both bounds)
            temp_low = torch.clamp(torch.log(torch.tensor(2.3, device=token_sims.device)) - torch.log(self.temperature), min=0) ** 4
            temp_high = torch.clamp(torch.log(self.temperature) - torch.log(torch.tensor(4.0, device=token_sims.device)), min=0) ** 4
            l_cal = temp_low + temp_high 

            reg_loss = ( 8.0 * l_cal + 0.15 * l_nonneg)
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
        visual_feats = self.visual_embedder(frames)
        audio_feats = self.audio_embedder(audio)
        
        if self.training:
            clip_sims, token_sims = self.compute_all_similarities(audio_feats, visual_feats)
            return self.compute_contrastive_loss(clip_sims, token_sims)
        else:
            token_sims = self.compute_similarity_matrix(audio_feats, visual_feats)
            return token_sims

if __name__ == "__main__":
    model = AudioVisualModel()
    batch_size = 4
    frames = torch.randn(batch_size, 3, 224, 224)
    audio = torch.randn(batch_size, 16331)
    loss = model(frames, audio)
    print(f"Training loss: {loss.item()}")
    model.eval()
    
    similarities = model(frames, audio)
    print(f"Inference similarities shape: {similarities.shape}")  # Should be (batch_size)
    print(f"Similarity values: {similarities}")
