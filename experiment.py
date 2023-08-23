import torch
from load_data import ShotDataset
import torch.nn as nn
import numpy as np
from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn, resnet18
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import torch.nn.functional as F
from pytorch_metric_learning import losses
from skorch import NeuralNetClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits, labels):
        #logits_normalized = F.normalize(logits, p=2, dim=1)
        logits = torch.div(
            torch.matmul(
                logits, torch.transpose(logits, 0, 1)
            ),
            self.temperature,
        )
        print(logits)
        print(labels)

        return losses.NTXentLoss(temperature=0.07)(torch.flatten(logits), labels)


class Experiment:
    
    def __init__(self, opt):

        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = vgg19_bn(pretrained=True)
        self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=5)
        #self.model = MovieShotModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam
        self.criterion = torch.nn.CrossEntropyLoss()
        self.CV_model = NeuralNetClassifier(self.model, criterion=self.criterion, optimizer=self.optimizer, lr=opt['lr'], batch_size=32)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=1e-6)
        #self.criterion = SupervisedContrastiveLoss(temperature=0.1)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-6)


    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_test_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_test_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_test_loss

    def train_iteration(self, dataset):
        #x, y = data
        #x = x.to(self.device)
        #y = y.to(self.device)

        #logits = self.model(x)
        #kfold = StratifiedKFold(n_splits=5, shuffle=True)
        #logits = cross_val_score(estimator=self.CV_model, X=torch.tensor(x), cv=kfold)

        #l2_lambda = 0.001
        #l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
        #loss = loss + l2_lambda * l2_norm
        
        kf = KFold(n_splits=5, shuffle=True)
        normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG-16 - ImageNet Normalization

        train_transform = T.Compose([
            T.Resize(256),
            T.ColorJitter(),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            normalize
        ])

        eval_transform = T.Compose([
            T.Resize(256),
            T.ToTensor(),
            normalize
        ])
        # Loop through each fold
        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
            print(f"Fold {fold + 1}")
            print("-------")

            # Define the data loaders for the current fold
            train_loader = DataLoader(
                dataset=ShotDataset(dataset, train_transform),
                batch_size=32,
                shuffle=True,
                sampler=torch.utils.data.SubsetRandomSampler(train_idx)
            )
            test_loader = DataLoader(
                dataset=ShotDataset(dataset, eval_transform),
                batch_size=32, 
                shuffle=False,
                sampler=torch.utils.data.SubsetRandomSampler(test_idx)
            )

            # Initialize the model and optimizer
            # Train the model on the current fold
            for epoch in range(1, 11):
                self.model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = nn.functional.nll_loss(output, target)
                    loss.backward()
                    self.optimizer.step()
            # Evaluate the model on the test set
            self.model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_loss += nn.functional.nll_loss(output, target, reduction="sum").item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            test_accuracy = correct / len(test_loader.dataset)

        return test_loss, test_accuracy

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        true_lables = []
        preds = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                true_lables.append(torch.Tensor.cpu(y))

                logits = self.model(x)
                loss += self.criterion(logits, y)
                pred = torch.argmax(logits, dim=-1)
                preds.append(torch.Tensor.cpu(pred))

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        true_lables = torch.Tensor(true_lables)
        preds = torch.Tensor(preds)
        f1 = f1_score(true_lables, preds, average='macro')
        #recall_score = recall_score(true_lables, preds, average='macro')
        #precision_score = precision_score(true_lables, preds, average='macro')
        cm = confusion_matrix(true_lables, preds)
        self.model.train()
        return mean_accuracy, mean_loss, f1, cm