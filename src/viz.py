import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class AudioVisualizer:
    def __init__(self, patch_size=14, image_size=224):
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = image_size // patch_size
        
        # (transparent -> blue -> red -> yellow)
        colors = [
            (0,0,0,0),     # Transparent for low attention
            (0,0,1,0.5),   # Blue for medium-low
            (1,0,0,0.7),   # Red for medium-high  
            (1,1,0,1)      # Yellow for high attention
        ]
        self.cmap = LinearSegmentedColormap.from_list('custom', colors)

    def _validate_inputs(self, frame, audio):
        """Validate that inputs are properly normalized"""
        frame_min, frame_max = frame.min().item(), frame.max().item()
        assert -3 <= frame_min <= 3, f"Frame min {frame_min} outside expected ImageNet normalized range"
        assert -3 <= frame_max <= 3, f"Frame max {frame_max} outside expected ImageNet normalized range"
        audio_min, audio_max = audio.min().item(), audio.max().item()
        assert -1 <= audio_min <= 1, f"Audio min {audio_min} outside normalized range"
        assert -1 <= audio_max <= 1, f"Audio max {audio_max} outside normalized range"
    
    def get_attention_maps(self, model, frame, audio):
        """Get attention maps for each audio token"""
        model.eval()
        with torch.no_grad():
            visual_feats = model.visual_embedder(frame)    # (1, Nv, D)
            audio_feats = model.audio_embedder(audio)      # (1, Na, D)
            
            similarity = model.compute_similarity_matrix(
                audio_feats, 
                visual_feats
            ).squeeze(0)  # (Na, Nv)
            
            attention_maps = self.patches_to_heatmaps(similarity)
            
        return attention_maps
    
    def patches_to_heatmaps(self, patch_attention):
        """Convert patch-level attention to pixel-level heatmaps"""
        Na, Nv = patch_attention.shape
        # Sq attn values to increase contrast
        patches = patch_attention.reshape(Na, self.num_patches, self.num_patches)
        patches = patches ** 2  
        heatmaps = F.interpolate(
            patches.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        return heatmaps
    
    def create_overlay_frame(self, frame: np.ndarray, heatmap: np.ndarray, alpha=0.5):
        """Create a single frame with heatmap overlay"""
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = np.power(heatmap, 2) 
        heatmap_colored = self.cmap(heatmap)
        heatmap_bgr = (heatmap_colored[...,:3] * 255).astype(np.uint8)
        overlay = ((1-alpha) * frame + alpha * heatmap_bgr).astype(np.uint8)
        return overlay
    
    def make_attention_video(self, model, frame, audio, output_path, video_path=None, fps=50):
        """Create attention visualization video - synchronized to 1s duration
        Args:
            frame: ImageNet normalized frame tensor [1, C, H, W]
            audio: normalized audio tensor [1, T] 
        """
        self._validate_inputs(frame, audio)
        attention_maps = self.get_attention_maps(model, frame, audio)
        frame_np = frame.squeeze(0).permute(1,2,0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
        frame_np = frame_np * std + mean
        frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_video_path = str(output_path.with_suffix('.temp.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            temp_video_path,
            fourcc,
            fps,
            (self.image_size, self.image_size)
        )
        for heatmap in attention_maps:
            overlay = self.create_overlay_frame(frame_np, heatmap.cpu().numpy())
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
        writer.release()
        if video_path is not None:
            import ffmpeg
            
            audio_input = ffmpeg.input(video_path).audio
            video_input = ffmpeg.input(temp_video_path).video
            
            stream = ffmpeg.output(
                video_input, 
                audio_input, 
                str(output_path),
                vcodec='copy',
                acodec='aac'
            ).overwrite_output()
            
            try:
                stream.run(capture_stdout=True, capture_stderr=True)
            except ffmpeg.Error as e:
                print('stdout:', e.stdout.decode('utf8'))
                print('stderr:', e.stderr.decode('utf8'))
                raise e
            
            Path(temp_video_path).unlink()
        else:
            Path(temp_video_path).rename(output_path)
            
    def plot_attention_snapshot(self, model, frame, audio, num_timesteps=5, axes=None, fig=None):
        """
        Plot attention maps at different timesteps
        
        Args:
            model: The audio-visual model
            frame: ImageNet normalized frame tensor [1, C, H, W]
            audio: Normalized audio tensor [1, T]
            num_timesteps: Number of timesteps to visualize
            axes: Optional matplotlib axes for subplot (array-like)
            fig: Optional matplotlib figure
        """
        self._validate_inputs(frame, audio)
        attention_maps = self.get_attention_maps(model, frame, audio)
        frame_np = frame.squeeze(0).permute(1,2,0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
        frame_np = frame_np * std + mean
        frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
        timesteps = np.linspace(0, len(attention_maps)-1, num_timesteps).astype(int)
        if axes is None:
            fig, axes = plt.subplots(1, num_timesteps, figsize=(2*num_timesteps, 4))
            created_fig = True
        else:
            created_fig = False
            
        if num_timesteps == 1:
            axes = [axes]
        for ax, t in zip(axes, timesteps):
            heatmap = attention_maps[t].cpu().numpy()
            overlay = self.create_overlay_frame(frame_np, heatmap)
            ax.imshow(overlay)
            ax.set_title(f'Time: {t/99:.1f}s')  # Assuming 998 frames for 10s
            ax.axis('off')
            
        if created_fig:
            plt.tight_layout()
            plt.savefig('outputs/attention_snapshot.png')
            plt.close()
        
        return fig if created_fig else None

if __name__ == "__main__":
    from model import AudioVisualModel
    model = AudioVisualModel().eval() 
    visualizer = AudioVisualizer()
    white_frame = torch.ones(1, 3, 224, 224)  # Start with ones
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    white_frame = (white_frame - mean) / std
    black_frame = torch.zeros(1, 3, 224, 224)  # Start with zeros
    black_frame = (black_frame - mean) / std
    t = torch.linspace(0, 2*torch.pi, 16331)
    audio = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440Hz tone
    
    print("Frame ranges:")
    print(f"White frame: min={white_frame.min():.3f}, max={white_frame.max():.3f}")
    print(f"Black frame: min={black_frame.min():.3f}, max={black_frame.max():.3f}")
    print(f"Audio range: min={audio.min():.3f}, max={audio.max():.3f}")
    
    print("\nCreating visualization with white background...")
    visualizer.make_attention_video(
        model, white_frame, audio,
        'test_attention_white.mp4'
    )
    visualizer.plot_attention_snapshot(
        model, white_frame, audio,
        num_timesteps=8
    )
    print("\nCreating visualization with black background...")
    visualizer.make_attention_video(
        model, black_frame, audio,
        'test_attention_black.mp4'
    )
    visualizer.plot_attention_snapshot(
        model, black_frame, audio,
        num_timesteps=8
    )
    print("\nDone! Check the output files.")