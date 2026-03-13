import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    def __init__(self, embed_size=512, train_cnn=False):
        super().__init__()

        resnet = models.resnet50(pretrained=True)

        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)

        self.fc = nn.Linear(2048, embed_size)

        self.train_cnn = train_cnn

    def forward(self, images):

        with torch.set_grad_enabled(self.train_cnn):
            features = self.cnn(images)

        B, C, H, W = features.size()
        features = features.view(B, C, H*W).permute(0, 2, 1)  # (B, 49, 2048)
        features = self.fc(features)  # (B, 49, embed_size)
        return features  # (B, 49, 512) if embed_size=512