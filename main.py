
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import shutil
import os
import jiwer
from torch.utils.data import DataLoader
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")

# Create Checkpoint Folder
save_path = '/content/drive/My Drive/SpeechInSight_Models'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# --- 2. DATA PROCESSING RECIPE ---
# Vocabulary
chars = "'" + " abcdefghijklmnopqrstuvwxyz"
char_to_num = {char: i + 1 for i, char in enumerate(chars)}
num_to_char = {i + 1: char for i, char in enumerate(chars)}


class TextTransform:
    def __init__(self):
        self.char_map = char_to_num
        self.index_map = num_to_char

    def text_to_int(self, text):
        int_list = []
        for c in text:
            if c == ' ':
                ch = ' '
            else:
                ch = c
            if ch in self.char_map:
                int_list.append(self.char_map[ch])
        return int_list

    def int_to_text(self, labels):
        string = []
        for i in labels:
            string.append(self.index_map.get(i, ""))
        return ''.join(string)


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

text_transform = TextTransform()


def data_processing(data):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for (waveform, _, utterance, _, _, _) in data:
        spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).transpose(1, 2)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


# --- 3. SMART DATA LOADING (Cache & Copy) ---
from torchaudio.datasets import LIBRISPEECH

local_data_path = './data'
drive_data_path = '/content/drive/My Drive/SpeechData_LibriSpeech'

print("🔄 Checking Data Source...")
if os.path.exists(drive_data_path):
    print(f"✅ Found dataset in Drive!")
    if not os.path.exists(local_data_path):
        print("⏳ Copying to local machine for speed...")
        shutil.copytree(drive_data_path, f"{local_data_path}/LibriSpeech")
    train_dataset = LIBRISPEECH(local_data_path, url="train-clean-100", download=False)
else:
    print("⚠️ Dataset not in Drive. Downloading...")
    if not os.path.exists(local_data_path):
        os.makedirs(local_data_path)
    train_dataset = LIBRISPEECH(local_data_path, url="train-clean-100", download=True)
    print("💾 Backing up to Google Drive...")
    shutil.copytree(f"{local_data_path}/LibriSpeech", drive_data_path)

train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, collate_fn=data_processing)
print("🚀 Data Loader Ready!")


# --- 4. MODEL & OPTIMIZER ---
class CRNN(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(CRNN, self).__init__()
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)
        self.rnn = nn.GRU(input_size=32 * n_feats, hidden_size=rnn_dim, num_layers=n_rnn_layers, batch_first=True,
                          bidirectional=True)
        self.classifier = nn.Linear(rnn_dim * 2, n_class)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        batch, time, channels, feats = x.size()
        x = x.view(batch, time, channels * feats)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        return x


model = CRNN(n_cnn_layers=1, n_rnn_layers=3, rnn_dim=512,
             n_class=len(char_to_num) + 1, n_feats=128).to(device)

criterion = nn.CTCLoss(blank=0).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4,
                                          steps_per_epoch=int(len(train_loader)),
                                          epochs=10, anneal_strategy='linear')


# --- 5. CHECKPOINT HELPERS ---
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    filename = f"{path}/checkpoint_epoch_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, filename)
    print(f"💾 Saved Checkpoint: {filename}")


def load_checkpoint(model, optimizer, scheduler, path):
    if not os.path.exists(path): return 1
    files = [f for f in os.listdir(path) if f.startswith('checkpoint_epoch_')]
    if not files: return 1
    files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    latest_file = files[-1]
    print(f"♻️ Found save file! Resuming from: {latest_file}")
    checkpoint = torch.load(f"{path}/{latest_file}")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'] + 1


# --- 6. TRAIN LOOP ---
def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, text_transform):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        spectrograms = spectrograms.unsqueeze(1)

        optimizer.zero_grad()
        output = model(spectrograms)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch_idx % 50 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(spectrograms)}/{data_len} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# Execution
start_epoch = load_checkpoint(model, optimizer, scheduler, save_path)
if start_epoch > 10:
    print("✅ Training already completed!")
else:
    print(f"🚀 Starting/Resuming Training from Epoch {start_epoch}...")
    for epoch in range(start_epoch, 11):
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch, text_transform)
        save_checkpoint(model, optimizer, scheduler, epoch, save_path)


# --- 7. FINAL EVALUATION (Decoder) ---
def greedy_decoder(model, device, test_loader, text_transform):
    model.eval()
    data = next(iter(test_loader))
    spectrograms, labels, input_lengths, label_lengths = data
    spectrograms, labels = spectrograms.to(device), labels.to(device)
    spectrograms = spectrograms.unsqueeze(1)

    output = model(spectrograms)
    arg_maxes = torch.argmax(output, dim=2)
    decodes, targets = [], []
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != 0:
                if j != 0 and index == args[j - 1]: continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
    return decodes, targets


print("\n--- 🔍 Final Evaluation ---")
test_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, collate_fn=data_processing)
decodes, targets = greedy_decoder(model, device, test_loader, text_transform)
for i in range(3):
    print(f"Target:    {targets[i]}")
    print(f"Predicted: {decodes[i]}")
    print("-" * 30)
print(f"📊 Final WER: {jiwer.wer(targets, decodes):.2f}")
