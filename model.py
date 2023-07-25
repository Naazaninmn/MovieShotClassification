import torch.nn as nn
from torchvision.models import resnet18


class FeatureExtractor( nn.Module ):
    def __init__(self):
        super( FeatureExtractor, self ).__init__()
        self.resnet18 = resnet18( pretrained=True )

    def forward(self, x):
        x = self.resnet18.conv1( x )
        x = self.resnet18.bn1( x )
        x = self.resnet18.relu( x )
        x = self.resnet18.maxpool( x )
        x = self.resnet18.layer1( x )
        x = self.resnet18.layer2( x )
        x = self.resnet18.layer3( x )
        x = self.resnet18.layer4( x )
        x = self.resnet18.avgpool( x )
        x = x.squeeze()
        if len( x.size() ) < 2:
            return x.unsqueeze( 0 )
        return x


class MovieShotModel( nn.Module ):
    def __init__(self):
        super( MovieShotModel, self ).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = nn.Linear( 512, 5 )

    def forward(self, x):
        x = self.feature_extractor( x )
        x = self.classifier( x )

        return x


    

