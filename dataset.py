import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
import numpy as np
import random
import av
from typing import Dict, List
import torch.nn as nn
import torchaudio.transforms as T
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
#import dataloader
from torch.utils.data import DataLoader
# Attempt to use fork for potentially faster dataloader start
#try:
    #multiprocessing.set_start_method('fork', force=True)
#except:
multiprocessing.set_start_method('spawn', force=True)
import gc
# Global normalization constants (ImageNet)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

def extract_audio_from_video(video_path: Path) -> torch.Tensor:
    """Extract entire 1s audio from video."""
    try:
        container = av.open(str(video_path))
        audio = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=16000)
        
        samples = []
        for frame in container.decode(audio):
            frame.pts = None
            frame = resampler.resample(frame)[0]
            samples.append(frame.to_ndarray().reshape(-1))
        container.close()

        samples = torch.tensor(np.concatenate(samples))
        samples = samples.float() / 32768.0  # Convert to float and normalize
        return samples
    except:
        print(f"Failed to load audio from {video_path}")
        return torch.zeros(16331)
    finally:
        if container:
            container.close()
        #gc.collect()
        #torch.cuda.empty_cache()

'''def load_and_preprocess_video(video_path: str, sample_fps: int) -> torch.Tensor:
    """Load only one random frame from the 1s video using PyAV, resize, and normalize."""
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    # Duration = 1s, get original fps
    original_fps = float(video_stream.average_rate)
    video_duration = 1.0
    num_original_frames = int(round(original_fps * video_duration))

    # Compute frame indices as before
    desired_frame_count = int(video_duration * sample_fps)  # equals sample_fps
    frame_indices = np.linspace(0, num_original_frames - 1, desired_frame_count, dtype=int)
    chosen_index = frame_indices[np.random.randint(0, desired_frame_count)]

    # Calculate PTS for chosen frame
    chosen_time_seconds = chosen_index / original_fps
    chosen_pts = int(chosen_time_seconds / video_stream.time_base)

    # Seek and decode that single frame
    container.seek(chosen_pts, any_frame=False, backward=True, stream=video_stream)
    decoded_frame = None
    for frame in container.decode(video_stream):
        decoded_frame = frame.to_rgb().to_ndarray()
        break
    container.close()
    #video_stream.codec_context.close()

    # Convert to tensor, resize, and normalize
    frame_tensor = torch.from_numpy(decoded_frame).permute(2, 0, 1).float() / 255.0
    frame_tensor = torch.nn.functional.interpolate(
        frame_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze(0)
    frame_tensor = (frame_tensor - IMAGENET_MEAN) / IMAGENET_STD

    return frame_tensor'''

def load_and_preprocess_video(video_path: str, sample_fps: int) -> torch.Tensor:
    """Load only one random frame from the 1s video using PyAV, resize, and normalize."""
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    original_fps = float(video_stream.average_rate)
    video_duration = 1.0
    num_original_frames = int(round(original_fps * video_duration))

    # Compute frame indices as before
    desired_frame_count = int(video_duration * sample_fps)  # equals sample_fps
    frame_indices = np.linspace(0, num_original_frames - 1, desired_frame_count, dtype=int)
    chosen_index = frame_indices[np.random.randint(0, desired_frame_count)]

    # Calculate PTS for chosen frame
    chosen_time_seconds = chosen_index / original_fps
    chosen_pts = int(chosen_time_seconds / video_stream.time_base)

    # Seek and find closest frame
    container.seek(chosen_pts, stream=video_stream, any_frame=False, backward=True)
    
    closest_frame = None
    min_pts_diff = float('inf')
    
    # Find frame closest to our target PTS
    for frame in container.decode(video_stream):
        pts_diff = abs(frame.pts - chosen_pts)
        
        if pts_diff < min_pts_diff:
            min_pts_diff = pts_diff
            closest_frame = frame
        
        # If we've gone too far past our target, stop
        if frame.pts > chosen_pts + original_fps/10:  # Allow 1/10th second overshoot
            break
    
    container.close()

    if closest_frame is None:
        raise ValueError(f"Failed to find appropriate frame for index {chosen_index}")
    
    # Convert to tensor, resize, and normalize
    decoded_frame = closest_frame.to_rgb().to_ndarray()
    frame_tensor = torch.from_numpy(decoded_frame).permute(2, 0, 1).float() / 255.0
    frame_tensor = torch.nn.functional.interpolate(
        frame_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze(0)
    frame_tensor = (frame_tensor - IMAGENET_MEAN) / IMAGENET_STD

    return frame_tensor

class VideoBatchSampler(Sampler):
    def __init__(self, vid_nums: List[int], batch_size: int):
        self.vid_nums = np.array(vid_nums)
        self.batch_size = batch_size
        self.total_samples = len(vid_nums)

    def __iter__(self):
        all_indices = list(range(self.total_samples))
        random.shuffle(all_indices)
        
        current_batch = []
        used_vids = set()
        
        for idx in all_indices:
            vid = self.vid_nums[idx]
            if vid not in used_vids:
                current_batch.append(idx)
                used_vids.add(vid)
                if len(current_batch) == self.batch_size:
                    yield current_batch
                    current_batch = []
                    used_vids = set()
        
        if current_batch:
            yield current_batch
    
    def __len__(self):
        return self.total_samples // self.batch_size

class AudioVisualDataset(Dataset):
    def __init__(self, data_root: str, sample_fps: int = 20):
        self.data_root = Path(data_root)
        self.sample_fps = sample_fps
        self.video_files = sorted(list(self.data_root.glob("*.mp4")))
        
        self.vid_to_files = {}
        for file in self.video_files:
            vid_num = int(file.stem.split('_')[0])
            if vid_num not in self.vid_to_files:
                self.vid_to_files[vid_num] = []
            self.vid_to_files[vid_num].append(file)
            
        self.vid_nums = [int(f.stem.split('_')[0]) for f in self.video_files]
        print("max of vid_nums: ", max(self.vid_nums))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
    
        video_path = self.video_files[idx]
        try:
            audio = extract_audio_from_video(video_path)
            video_frame = load_and_preprocess_video(str(video_path), self.sample_fps)
            return {
                'video_path': str(video_path),
                'video_frames': video_frame, 
                'audio': audio,
                'vid_num': int(video_path.stem.split('_')[0]),
                'segment_num': int(video_path.stem.split('_')[1]),
            }
        except Exception as e:
            print(f"Error processing {self.video_files[idx]}: {str(e)}")
            return {
                'video_path': str(self.video_files[idx]),
                'video_frames': torch.zeros(3, 224, 224),
                'audio': torch.zeros(16331),
                'vid_num': -1,
                'segment_num': -1
            }

def collate_fn(batch):
    video_tokens = torch.stack([item['video_frames'] for item in batch])
    max_audio_len = max(item['audio'].shape[0] for item in batch)
    audio_padded = torch.zeros(len(batch), max_audio_len)
    for i, item in enumerate(batch):
        audio_len = item['audio'].shape[0]
        audio_padded[i, :audio_len] = item['audio']
    
    return {
        'frame': video_tokens,
        'audio': audio_padded,
        'vid_nums': [item['vid_num'] for item in batch],
        'segment_nums': [item['segment_num'] for item in batch],
        'video_paths': [str(item['video_path']) for item in batch]
    }

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataset = AudioVisualDataset(
        data_root="/home/cisco/nvmefudge/vggsound_1seconds",
        sample_fps=20
    )

    batch_sampler = VideoBatchSampler(
        vid_nums=dataset.vid_nums,
        batch_size=48
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=16,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=6
    )
    i=0
    while True:  # Just keep going like in training
        pbar = tqdm(dataloader, desc=f'Epoch {i}')  
        for batch in pbar:
            pass  # Just iterate through, simulating data being sent to training
        i+=1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    import unittest
    
    class DatasetTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            # Initialize dataset and dataloader once for all tests
            cls.dataset = AudioVisualDataset(
                "/home/cisco/nvmefudge/vggsound_1seconds",  # Update path as needed
                sample_fps=20
            )
            
            cls.batch_size = 4
            cls.batch_sampler = VideoBatchSampler(
                vid_nums=cls.dataset.vid_nums,
                batch_size=cls.batch_size
            )
            
            cls.dataloader = DataLoader(
                cls.dataset,
                batch_sampler=cls.batch_sampler,
                num_workers=0,
                collate_fn=collate_fn
            )
            
            # Get one batch for testing
            cls.batch = next(iter(cls.dataloader))

        def test_batch_structure(self):
            """Test if batch contains all required keys with correct types"""
            required_keys = {'frame', 'audio', 'vid_nums', 'segment_nums', 'video_paths'}
            self.assertEqual(set(self.batch.keys()), required_keys, "Batch missing required keys")
            
            # Check types
            self.assertIsInstance(self.batch['frame'], torch.Tensor, "Frames should be a tensor")
            self.assertIsInstance(self.batch['audio'], torch.Tensor, "Audio should be a tensor")
            self.assertIsInstance(self.batch['vid_nums'], list, "Video numbers should be a list")
            self.assertIsInstance(self.batch['segment_nums'], list, "Segment numbers should be a list")
            self.assertIsInstance(self.batch['video_paths'], list, "Video paths should be a list")
            self.assertIsInstance(self.batch['video_paths'][0], str, "Video paths should be strings")

        def test_video_frame_format(self):
            """Test video frame tensor format and normalization"""
            frames = self.batch['frame']
            
            # Check shape
            self.assertEqual(len(frames.shape), 4, "Frames should be 4D tensor")
            self.assertEqual(frames.shape[0], self.batch_size, "Incorrect batch size")
            self.assertEqual(frames.shape[1], 3, "Frames should have 3 channels")
            self.assertEqual(frames.shape[2], 224, "Frame height should be 224")
            self.assertEqual(frames.shape[3], 224, "Frame width should be 224")
            
            # Check ImageNet normalization (approximately)
            means = frames.mean(dim=[0,2,3])
            stds = frames.std(dim=[0,2,3])
            
            # ImageNet normalized data should have approximately these ranges
            self.assertTrue(torch.all(frames >= -3), "Frames min value out of expected range")
            self.assertTrue(torch.all(frames <= 3), "Frames max value out of expected range")
            
            # Rough check of channel means and stds (should be approximately 0 and 1 after normalization)
            for i, (mean, std) in enumerate(zip(means, stds)):
                self.assertGreater(std.item(), 0.1, f"Channel {i} has suspiciously low standard deviation")
                self.assertLess(abs(mean.item()), 0.5, f"Channel {i} mean too far from 0")

        def test_audio_format(self):
            """Test audio tensor format and normalization"""
            audio = self.batch['audio']
            
            # Check shape
            self.assertEqual(len(audio.shape), 2, "Audio should be 2D tensor (batch, time)")
            self.assertEqual(audio.shape[0], self.batch_size, "Incorrect batch size")
            
            # Check normalization
            self.assertTrue(torch.all(audio >= -1), "Audio values should be >= -1")
            self.assertTrue(torch.all(audio <= 1), "Audio values should be <= 1")
            
            # Check if audio length is reasonable (expect ~1 second at 16kHz)
            self.assertGreater(audio.shape[1], 15000, "Audio seems too short for 1s at 16kHz")
            self.assertLess(audio.shape[1], 17000, "Audio seems too long for 1s at 16kHz")

        def test_batch_sampling(self):
            """Test if VideoBatchSampler provides unique videos per batch"""
            # Check if all video numbers in batch are unique
            unique_vids = set(self.batch['vid_nums'])
            self.assertEqual(len(unique_vids), len(self.batch['vid_nums']), 
                           "Batch contains duplicate videos")
            self.assertEqual(len(self.batch['vid_nums']), self.batch_size, 
                           "Batch size mismatch")
            
            # All metadata lists should have same length
            self.assertEqual(len(self.batch['vid_nums']), len(self.batch['segment_nums']),
                           "Metadata length mismatch")
            self.assertEqual(len(self.batch['vid_nums']), len(self.batch['video_paths']),
                           "Metadata length mismatch")

        def test_video_path_validity(self):
            """Test if video paths exist and are properly formatted"""
            for path in self.batch['video_paths']:
                self.assertTrue(Path(path).exists(), f"Video file does not exist: {path}")
                self.assertTrue(path.endswith('.mp4'), "Video should be MP4 format")
                
                # Check path format matches expected pattern (e.g., "vid_num_segment_num.mp4")
                filename = Path(path).stem
                parts = filename.split('_')
                self.assertEqual(len(parts), 2, "Video filename should have format 'vid_num_segment_num'")
                self.assertTrue(parts[0].isdigit(), "Video number should be integer")
                self.assertTrue(parts[1].isdigit(), "Segment number should be integer")
        
       

    if __name__ == "__main__":
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
        '''