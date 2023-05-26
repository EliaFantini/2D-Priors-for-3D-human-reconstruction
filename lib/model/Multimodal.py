import torch.nn as nn

class CLIP_Transform(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Project to a lower resolution feature map
        self.fc = nn.Linear(hidden_dim, 256*8*8)
        self.upsample = nn.Upsample((128, 128), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.fc(x)
        # Reshape to [bz, 256, 8, 8]
        x = x.view(-1, 256, 8, 8)
        # Upsample to [bz, 256, 128, 128]
        x = self.upsample(x)
        return x
    
