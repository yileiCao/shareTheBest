import torch
from torchvision import models


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
MODEL_STATE = "/Users/yileicao/Documents/Birmingham_project/summer_project/model_parameter/epoch-34.pth"
# MODEL_STATE = "/Users/yileicao/Documents/Birmingham_project/summer_project/model_para_test/mnv3_best_64.pth"
MODELS = {
    "vgg16": (models.resnet18, 25088),
    "mobileNetV2": (models.mobilenet_v2, 1280),
    "mobileNetV3Small": (models.mobilenet_v3_small, 576)
}

'''
class Nimanet(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()

        self.base_model, self.num_features = MODELS[base_model]
        self.features = self.base_model(pretrained=True).features
        self.classifier = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0),
            torch.nn.Linear(self.num_features, 10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        if self.base_model == models.mobilenet_v2:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        buckets = torch.arange(1, 11).unsqueeze(0)
        mu = (buckets * x).sum(axis=1)
        # std = torch.sqrt(torch.sum(((buckets - mu) ** 2) * x, 1))
        std = torch.sqrt(torch.sum(((buckets - mu.unsqueeze(1).repeat(1, 10)) ** 2) * x, 1))

        return [mu, std]
'''


class Nimanet(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()

        self.model, self.num_features = MODELS[base_model]
        tem_model = self.model(pretrained=True)
        self.base_model = torch.nn.Sequential(*list(tem_model.children())[:-1])
        self.head = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0),
            torch.nn.Linear(self.num_features, 10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        if self.model == models.mobilenet_v2:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.head(x)

        buckets = torch.arange(1, 11).unsqueeze(0)
        mu = (buckets * x).sum(axis=1)
        # std = torch.sqrt(torch.sum(((buckets - mu) ** 2) * x, 1))
        std = torch.sqrt(torch.sum(((buckets - mu.unsqueeze(1).repeat(1, 10)) ** 2) * x, 1))

        return [mu, std]

