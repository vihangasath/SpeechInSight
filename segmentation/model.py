import torch
import torch.nn as nn
import torchaudio
import os

# --- 1. CONFIGURATION ---
class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        chars = "'" + " abcdefghijklmnopqrstuvwxyz"
        self.char_map = {char: i + 1 for i, char in enumerate(chars)}
        self.index_map = {i + 1: char for i, char in enumerate(chars)}

    def int_to_text(self, labels):
        string = []
        for i in labels:
            string.append(self.index_map.get(i, ""))
        return ''.join(string)

# --- 2. MODEL ARCHITECTURE ---
class CRNN(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(CRNN, self).__init__()
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)
        self.rnn = nn.GRU(input_size=32 * n_feats, 
                          hidden_size=rnn_dim, 
                          num_layers=n_rnn_layers, 
                          batch_first=True, 
                          bidirectional=True)
        self.classifier = nn.Linear(rnn_dim * 2, n_class)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        batch, time, channels, feats = x.size()
        x = x.view(batch, time, channels*feats) 
        x, _ = self.rnn(x)
        x = self.classifier(x)
        return x

# --- 3. HELPER FUNCTION TO LOAD MODEL ---
def load_model(path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the empty shell
    model = CRNN(n_cnn_layers=1, n_rnn_layers=3, rnn_dim=512, 
                 n_class=29, n_feats=128).to(device)
    
    if not os.path.exists(path):
        print(f"⚠️ Warning: Model file {path} not found!")
        return None, device

    # Load weights
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

# --- 4. PREDICT FUNCTION ---
def transcribe_clip(audio_path, model, device):
    text_transform = TextTransform()
    test_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)

    try:
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        
        # Resample to 16k
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Mono check
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        spec = test_audio_transforms(waveform)
        spec = spec.unsqueeze(0).to(device) # Add batch dim: (1, 1, Freq, Time)

        with torch.no_grad():
            output = model(spec)
            arg_maxes = torch.argmax(output, dim=2)

        # Decode
        decodes = []
        for args in arg_maxes:
            decode = []
            for j, index in enumerate(args):
                if index != 0:
                    if j != 0 and index == args[j - 1]:
                        continue
                    decode.append(index.item())
            decodes.append(text_transform.int_to_text(decode))
        
        return decodes[0]
        
    except Exception as e:
        return f"Error: {str(e)}"