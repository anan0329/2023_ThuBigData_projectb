import torch
from torch import nn
import timm

class SPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.pool2 = nn.MaxPool2d(kernel_size=7, stride=1, padding=7 // 2)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        return torch.cat([x, x1, x2, x3], dim=1)

class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=hidden_units*53*53,
                    out_features=output_shape), 
        #   nn.Sigmoid(), 
          nn.LogSoftmax(dim=1)
      )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        return x
    
resnet_34 = timm.create_model('resnet34', pretrained=False, num_classes=0)
# resnet_34.fc = nn.Sequential(
#     nn.Linear(in_features=512, out_features=2), 
#     nn.LogSoftmax(dim=1)
# )
beitv2_224 = timm.create_model('beitv2_large_patch16_224.in1k_ft_in22k_in1k', pretrained=False, num_classes=0)
# beitv2_224.head = nn.Sequential(
#     nn.Linear(in_features=1024, out_features=2), 
#     nn.LogSoftmax(dim=1)
# )
    
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()
        


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arr = torch.randn([32, 3, 224, 224]).to(device)
    model = TinyVGG(
        input_shape=3,
        hidden_units=10, 
        output_shape=2
    ).to(device)
    print(model(arr).shape)
    
