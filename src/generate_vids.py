from pathlib import Path
import torch
from model import AudioVisualModel
from viz import AudioVisualizer
from dataset import extract_audio_from_video, load_and_preprocess_video
import random
import warnings
warnings.filterwarnings("ignore")

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory"""
    checkpoints = list(Path(output_dir).glob('checkpoint_epoch*.pt'))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: (
        int(str(x).split('epoch')[1].split('_')[0]),  
        int(str(x).split('step')[1].split('.')[0])   
    ))
    return latest

def load_model(checkpoint_path=None, output_dir=None, device='cuda'):
    """Load model from checkpoint"""
    model = AudioVisualModel().to(device)
    if checkpoint_path is None and output_dir is not None:
        checkpoint_path = find_latest_checkpoint(output_dir)
        if checkpoint_path is None:
            raise ValueError(f"No checkpoints found in {output_dir}")
    
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model

def generate_video(model, video_path, output_path, fps=50, device='cuda'):
    """Generate attention visualization video for a single video"""
    audio = extract_audio_from_video(video_path).to(device)
    video_frames = load_and_preprocess_video(str(video_path), sample_fps=20).to(device)
    audio = audio.unsqueeze(0)
    video_frames = video_frames.unsqueeze(0)
    visualizer = AudioVisualizer()
    visualizer.make_attention_video(
        model,
        video_frames,
        audio,
        output_path,
        video_path=str(video_path),
        fps=fps
    )
    print(f"Generated visualization: {output_path}")

def process_videos(
    video_paths=None,
    video_dir=None,
    num_random=5,
    output_dir='./viz_outputs',
    checkpoint_path=None,
    checkpoint_dir='./outputs',
    fps=50,
    device='cuda'
):
    """Process multiple videos and generate visualizations
    
    Args:
        video_paths: List of paths to specific videos to process
        video_dir: Directory containing videos to sample from
        num_random: Number of random videos to sample if using video_dir
        output_dir: Directory to save visualizations
        checkpoint_path: Path to specific checkpoint
        checkpoint_dir: Directory containing checkpoints (uses latest if no specific checkpoint)
        fps: Frames per second for output videos
        device: Device to run model on ('cuda' or 'cpu')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint_path, checkpoint_dir, device)
    if video_paths:
        videos_to_process = [Path(p) for p in video_paths]
    elif video_dir:
        all_videos = list(Path(video_dir).glob('*.mp4'))
        videos_to_process = random.sample(all_videos, min(num_random, len(all_videos)))
    else:
        raise ValueError("Must provide either video_paths or video_dir")
    for video_path in videos_to_process:
        output_path = output_dir / f'attention_{video_path.stem}.mp4'
        try:
            generate_video(model, video_path, output_path, fps, device)
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            continue

if __name__ == "__main__":
    # Option 1: Process specific videos
    video_list = [
        '/path/to/video1.mp4',
        '/path/to/video2.mp4'
    ]
    
    process_videos(
        video_paths=video_list,
        output_dir='./my_visualizations',
        device='cuda'  # or 'cpu' if no GPU available
    )
    
    # Option 2: Process random videos from a directory
    process_videos(
        video_dir='/path/to/video/directory',
        num_random=5,
        output_dir='./random_visualizations',
        checkpoint_path='./outputs/specific_checkpoint.pt',  # optional
        device='cuda'
    )