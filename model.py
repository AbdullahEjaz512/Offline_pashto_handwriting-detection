import torch
import torch.nn as nn

class PashtoCRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) for Offline Pashto Handwriting Detection.
    Expects input tensors of shape [Batch, 1, 32, Width].
    """
    def __init__(self, num_classes, hidden_size=256):
        super(PashtoCRNN, self).__init__()
        
        # 1. CNN Feature Extractor
        # Input shape: [B, 1, 32, W]
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [B, 64, 16, W/2]
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [B, 128, 8, W/4]
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Pool height by 2, but keep width the same (to preserve sequence length)
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # -> [B, 256, 4, W/4]
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Pool height by 2, keep width same
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # -> [B, 512, 2, W/4]
            
            # Block 5 (Final Conv to reduce height to 1)
            nn.Conv2d(512, 512, kernel_size=2, padding=0),    # valid padding, height 2->1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)                             # -> [B, 512, 1, (W/4)-1]
        )
        
        # 3. RNN Sequence Modeler
        # input_size must match the out_channels of the final Conv layer (512)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 4. Fully Connected (Classifier)
        # Bidirectional LSTM concatenates hidden states, so output is 2 * hidden_size
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Forward pass.
        Expects: x of shape [Batch, Channels=1, Height=32, Width=W]
        """
        # Step 1: Feature Extraction
        conv_out = self.cnn(x)  # Shape: [Batch, 512, 1, reduced_Width]
        
        # Step 2: Map to Sequence
        # Squeeze the height dimension (which is now 1)
        # Shape: [Batch, 512, reduced_Width]
        seq = conv_out.squeeze(2)
        
        # Permute to put sequence length in the middle (expected by typical PyTorch setups or batch_first=True)
        # Shape: [Batch, reduced_Width, 512]
        seq = seq.permute(0, 2, 1)
        
        # Step 3: RNN Sequence Modeling
        # rnn_out shape: [Batch, reduced_Width, hidden_size * 2]
        rnn_out, _ = self.rnn(seq)
        
        # Step 4: Classifier
        # Shape: [Batch, reduced_Width, num_classes]
        # We do NOT apply Softmax here, as CTCLoss expects raw logits (usually paired with LogSoftmax internally or applied via parameter)
        logits = self.fc(rnn_out)
        
        return logits

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    # Suppose our vocabulary has 50 unique characters + 3 special tokens (BLANK, PAD, UNK) = 53 classes
    num_classes = 53
    
    # Instantiate the model
    model = PashtoCRNN(num_classes=num_classes)
    
    # Create a dummy input tensor: Batch Size = 4, Channels = 1 (Grayscale), Height = 32, Width = 200
    dummy_input = torch.randn(4, 1, 32, 200)
    
    print(f"Input shape:  {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape} -> [Batch, Sequence_Length, Num_Classes]")
    print("\nModel is ready for training with CTCLoss!")
