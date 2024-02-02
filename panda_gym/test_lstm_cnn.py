import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101

class CNNLSTM(nn.Module):

    def __init__(
        self,
        features_dim: int = 512,
    ) -> None:

        super().__init__()
        # We assume CxHxW images (channels first)
        # self.fc_lstm = nn.Linear(16384, 512)
        self.fc_lstm = nn.Linear(51200, 512) # for deeper cnn
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=3)
        n_input_channels = 6 # last channel is feature channel from raw data [B, C H, W]
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=6, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = None
        for t in range(observations.size(1)):
            x = self.cnn(observations[:, t, :, :, :]) 
            x = self.fc_lstm(x)
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        return x
    

lstm_net = CNNLSTM().cuda()
x = torch.randn(512,10,6,160,160).cuda()
y = lstm_net(x)