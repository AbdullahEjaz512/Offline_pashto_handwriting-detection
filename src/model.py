import torch
import torch.nn as nn

class PashtoCRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) for Pashto Text Recognition.
    Configured for a default vocabulary size of 181 classes (including BLANK, PAD, UNK).
    Expects input tensors of shape [Batch, 1, 32, Width].
    """
    def __init__(self, num_classes=181, hidden_size=256):
        super(PashtoCRNN, self).__init__()
        
        # 1. Feature Extractor (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [B, 64, 16, W/2]
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [B, 128, 8, W/4]
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # -> [B, 256, 4, W/4]
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # -> [B, 512, 2, W/4]
            
            nn.Conv2d(512, 512, kernel_size=2, padding=0),    
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)                             # -> [B, 512, 1, (W/4)-1]
        )
        
        # 2. Sequence Modeler (BiLSTM)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Classifier (Dense)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv_out = self.cnn(x)
        # Squeeze height and permute to [Batch, SeqLength, Features]
        seq = conv_out.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(seq)
        logits = self.fc(rnn_out)
        return logits
