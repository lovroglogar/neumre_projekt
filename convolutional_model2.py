from torch import nn


class ConvolutionalModel3(nn.Module):
    def __init__(self):
        super(ConvolutionalModel3, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Dropout(p=0.25),

            nn.Linear(256, 32),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=32),
            nn.Dropout(p=0.25),

            #nn.Linear(128, 32),
            #nn.ReLU(),
            #nn.BatchNorm1d(num_features=32),
            #nn.Dropout(p=0.25),

            nn.Linear(32, 7)
        )

    def forward(self, x):
        return self.net(x)

