import torch.nn as nn
from torchvision.models import vgg16_bn


class VGG16( nn.Module ):
    def __init__(self):
        super( VGG16, self ).__init__()
        self.vgg16 = vgg16_bn( pretrained=True )
        self.vgg16.fc2 = nn.Sequential(nn.Linear(4096, 5))

    def forward(self, x):
        x = self.vgg16.layer1( x )
        x = self.vgg16.layer2( x )
        x = self.vgg16.layer3( x )
        x = self.vgg16.layer4( x )
        x = self.vgg16.layer5( x )
        x = self.vgg16.layer6( x )
        x = self.vgg16.layer7( x )
        x = self.vgg16.layer8( x )
        x = self.vgg16.layer9( x )
        x = self.vgg16.layer10( x )
        x = self.vgg16.layer11( x )
        x = self.vgg16.layer12( x )
        x = self.vgg16.layer13( x )
        x = self.vgg16.fc(x)
        x = self.vgg16.fc1(x)
        x = self.vgg16.fc2(x)
        return x

