import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                      padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        b, c, _, _ = x.size()

        # Channel Attention
        avg = self.avg_pool(x).view(b, c)
        max = self.max_pool(x).view(b, c)
        channel_att = self.sigmoid(self.mlp(avg) + self.mlp(max)).view(b, c, 1, 1)
        x = x * channel_att

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        return x

class EfficientNetB4_CBAM(nn.Module):
    def __init__(self, model, cbam_dropout_rate, classifier_dropout_rate, num_class):
        super().__init__()
        self.cbam = CBAM(160)
        self.cbam_dropout = nn.Dropout2d(p=cbam_dropout_rate) # regulate the effect of CBAM and prevent over-reliance on a single channel
        self.model = model
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.conv_head = model.conv_head
        self.bn2 = model.bn2
        self.blocks = nn.ModuleList(model.blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_dropout = nn.Dropout(p=classifier_dropout_rate)
        self.linear = nn.Linear(1792, num_class)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        for i in range(5):
            x = self.blocks[i](x)

        x = self.cbam(x)
        x = self.cbam_dropout(x)

        for i in range(5, len(self.blocks)):
            x = self.blocks[i](x)

        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.avg_pool(x) # reduce the dimensions of all feature maps into 1*1
        x = torch.flatten(x, 1) # concatenate all 1*1 feature maps
        x = self.classifier_dropout(x)
        x = self.linear(x)
        return x

    def gradient_control(self, module, state):
        for p in module.parameters():
            p.requires_grad = state

    def freeze(self, block_index, freeze_stem=True):
        if freeze_stem:
            self.gradient_control(self.conv_stem, False)
            self.gradient_control(self.bn1, False)
            for m in self.bn1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        for i in range(block_index + 1):
            self.gradient_control(self.blocks[i], False)
            for m in self.blocks[i].modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def unfreeze(self, block_index, unfreeze_stem=True):
        if unfreeze_stem:
            self.gradient_control(self.conv_stem, True)
            self.gradient_control(self.bn1, True)
            for m in self.bn1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

        if block_index is not None:
            self.gradient_control(self.blocks[block_index], True)
            for m in self.blocks[block_index].modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()


