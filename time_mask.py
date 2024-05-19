import os
import torchaudio
from torchaudio import transforms
import random

def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform, sample_rate

def apply_time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    num_freq, num_frames = cloned.shape[1], cloned.shape[2]
    
    for _ in range(num_masks):
        t = random.randrange(0, min(T, num_frames))
        t_zero = random.randrange(0, num_frames - t)
        mask_end = t_zero + t
        if replace_with_zero:
            cloned[0, :, t_zero:mask_end] = 0
        else:
            cloned[0, :, t_zero:mask_end] = cloned.mean(dim=2, keepdim=True).expand_as(cloned[0, :, t_zero:mask_end])
    return cloned

def save_masked_audio(masked_spec, sample_rate, output_path):
    # Griffin-Lim to convert the spectrogram back to audio
    griffin_lim = transforms.GriffinLim(n_fft=1024, win_length=512, hop_length=256)
    reconstructed_waveform = griffin_lim(masked_spec)

    # Save the reconstructed waveform
    torchaudio.save(output_path, reconstructed_waveform, sample_rate)
    print(f"Saved masked audio to {output_path}")

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            audio_path = os.path.join(directory, filename)
            waveform, sr = load_audio(audio_path)
            
            # Convert to spectrogram
            transform = transforms.Spectrogram(n_fft=1024, win_length=512, hop_length=256)
            spectro = transform(waveform)
            
            # Apply time mask
            masked_spectro = apply_time_mask(spectro, T=100, num_masks=1, replace_with_zero=True)
            
            # Construct the output file path
            output_path = os.path.splitext(audio_path)[0] + '_masked.wav'
            
            # Save the masked audio
            save_masked_audio(masked_spectro, sr, output_path)

if __name__ == "__main__":
    directory = 'H:/Bach/dcase2023_task2_baseline_ae/data/dcase2023t2/dev_data/raw/valve/train'
    process_directory(directory)
