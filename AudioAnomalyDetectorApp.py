import os
import torch
import torchaudio
import tkinter as tk
from tkinter import filedialog, messagebox
import common as com  # Make sure this module is available in your project
from networks.dcase2023t2_ae.dcase2023t2_ae import DCASE2023T2AE  # Import the correct class
from torch.utils.data import Dataset, DataLoader
import csv
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, folder_path, n_mels=128, frames=5, n_fft=1024, hop_length=512, transform=None):
        self.folder_path = 'C:/Users/Ahmed/OneDrive/Desktop/demo bavh'  # Path to the folder containing audio files
        self.audio_files = [os.path.join(self.folder_path, file) for file in os.listdir(self.folder_path) if file.endswith('.wav')]
        self.transform = transform
        self.n_mels = n_mels
        self.frames = frames
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )(waveform)

        # Ensure the feature vector is the expected shape
        num_frames = mel_spectrogram.size(2)
        if num_frames < self.frames:
            pad_amount = self.frames - num_frames
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, pad_amount))
        elif num_frames > self.frames:
            mel_spectrogram = mel_spectrogram[:, :, :self.frames]

        mel_spectrogram = torch.flatten(mel_spectrogram, start_dim=0)  # Use flatten with start_dim=0

        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)

        y_true = torch.tensor(0)
        condition = torch.zeros(1)

        return mel_spectrogram, y_true, condition, os.path.basename(audio_path)

class AudioAnomalyDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Anomaly Detection")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = self.load_args()

        self.n_mels = self.args.n_mels
        self.frames = self.args.frames

        self.args.model_path = os.path.join(os.getcwd(), 'models', 'saved_model', 'baseline', 'DCASE2023T2-AE_DCASE2023T2valve_id(0_)_seed13711.pth')

        self.args.eval_folder = os.path.join(os.getcwd(), 'data', 'dcase2023t2', 'dev_data', 'raw', 'valve', 'test')
        self.args.output_folder = os.path.join(os.getcwd(), 'eval_results')
        self.args.score = 'MAHALA'

        self.args.cuda = self.args.use_cuda and torch.cuda.is_available()
        self.args.gpu_id = [0]
        self.args.dataset = 'DCASE2023T2valve'
        self.args.device = self.device

        self.args.dev = True
        self.args.eval = not self.args.dev

        if not hasattr(self.args, 'section_ids'):
            self.args.section_ids = [0]

        device = self.args.device
        del self.args.device

        self.model = DCASE2023T2AE(self.args, train=False, test=True)

        self.args.device = device

        self.model.model.load_state_dict(torch.load(self.args.model_path, map_location=self.device))
        self.model.model.to(self.device)
        self.model.model.eval()

        if not hasattr(self.model, 'cov_source'):
            self.model.cov_source = torch.zeros((self.n_mels * self.frames, self.n_mels * self.frames), device=self.device)
        if not hasattr(self.model, 'cov_target'):
            self.model.cov_target = torch.zeros((self.n_mels * self.frames, self.n_mels * self.frames), device=self.device)

        self.setup_gui()

    def load_args(self):
        parser = com.get_argparse()
        param = com.yaml_load()
        flat_param = com.param_to_args_list(params=param)
        args = parser.parse_args(args=flat_param)
        args = parser.parse_args(namespace=args)
        return args

    def setup_gui(self):
        tk.Label(self.master, text="Audio Folder:").grid(row=0)
        self.audio_folder_entry = tk.Entry(self.master, width=50)
        self.audio_folder_entry.grid(row=0, column=1)
        tk.Button(self.master, text="Browse", command=self.browse_folder).grid(row=0, column=2)
        tk.Button(self.master, text="Detect Anomaly", command=self.detect_anomaly).grid(row=1, column=1)
        self.result_label = tk.Label(self.master, text="", fg="red")
        self.result_label.grid(row=2, column=1)

    def browse_folder(self):
        foldername = filedialog.askdirectory()
        self.audio_folder_entry.delete(0, tk.END)
        self.audio_folder_entry.insert(0, foldername)

    def create_test_loader(self, eval_folder):
        dataset = CustomDataset(
            eval_folder,
            n_mels=self.args.n_mels,
            frames=self.args.frames,
            n_fft=self.args.n_fft,
            hop_length=self.args.hop_length
        )
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        return test_loader

    def detect_anomaly(self):
        audio_folder = self.audio_folder_entry.get()
        self.args.eval_folder = audio_folder
        
        test_loader = self.create_test_loader(self.args.eval_folder)

        # Collect all anomaly scores to determine a suitable threshold
        all_anomaly_scores = []

        # CSV file to save the results
        results_csv = os.path.join(self.args.output_folder, 'anomaly_detection_results.csv')
        with open(results_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Filename", "Anomaly"])

            for batch in test_loader:
                data, _, _, basename = batch
                data = data.to(self.device).float()
                recon_data, _ = self.model.model(data)

                anomaly_score = torch.nn.functional.mse_loss(recon_data, data).item()
                all_anomaly_scores.append(anomaly_score)
                

                is_anomalous = 1 if anomaly_score > 0.9 else 0
                writer.writerow([basename[0], is_anomalous])

        # Adjust threshold based on the collected anomaly scores
        mean_score = np.mean(all_anomaly_scores)
        std_score = np.std(all_anomaly_scores)
        adjusted_threshold = (mean_score + 2 * std_score)-2 # For example, use mean + 2*std as the threshold
        

        # Re-evaluate and update results based on the adjusted threshold
        with open(results_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Filename", "Anomaly"])

            for batch in test_loader:
                data, _, _, basename = batch
                data = data.to(self.device).float()
                recon_data, _ = self.model.model(data)

                anomaly_score = torch.nn.functional.mse_loss(recon_data, data).item()
                is_anomalous = 1 if anomaly_score > adjusted_threshold else 0
                writer.writerow([basename[0], is_anomalous])

        # Read the results and update the GUI
        with open(results_csv, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            results = [row for row in reader]

        results_text = "\n".join([f"{filename}: {'Anomalous' if int(anomaly) == 1 else 'Normal'}" for filename, anomaly in results])
        self.result_label.config(text=f"Detection Complete:\n{results_text}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnomalyDetectorApp(root)
    root.mainloop()
