import torch
from load_data import ShotDataset
import torch.nn as nn
import numpy as np
from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn, resnet50, resnet101
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


class Experiment:
    
    def __init__(self, opt):

        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = vgg19(pretrained=True)
        self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=5)
        # self.model = resnet101(pretrained=True)
        # self.model.fc.out_features = 5
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True
        
        # freezing the first four layers
        for i in range(4):
            for param in self.model.features[i].parameters():
                param.requires_grad = False

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion = torch.nn.CrossEntropyLoss()


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
        total_test_loss = checkpoint['total_test_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_test_loss

    def train_iteration(self, train_examples, X, Y):
        
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=True)

        normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG-16 - ImageNet Normalization

        # training folds transformations
        train_transform = T.Compose([
            T.Resize(256),
            T.ColorJitter(),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            normalize
        ])

        # validation fold transformations
        eval_transform = T.Compose([
            T.Resize(256),
            T.ToTensor(),
            normalize
        ])
        
        # Loop through each fold
        for fold, (train_idx, test_idx) in enumerate(kf.split(X, Y)):
            print(f"Fold {fold + 1}")
            print("-------")
            
            # Defining the data loaders for the current fold
            train_loader = DataLoader(
                dataset=ShotDataset(train_examples, train_transform),
                sampler=torch.utils.data.SubsetRandomSampler(train_idx)
            )
            val_loader = DataLoader(
                dataset=ShotDataset(train_examples, eval_transform),
                sampler=torch.utils.data.SubsetRandomSampler(test_idx)
            )

            # Training the model on the current fold
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                # L2 regularization
                # l2_lambda = 0.01
                # l2_lambda = 0.001
                # l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
                # loss = loss + l2_lambda * l2_norm

                # L1 regularization 
                # l1_lambda = 0.01
                #l1_lambda = 0.001
                #l1_norm = sum(abs(p).sum() for p in self.model.parameters())
                #loss = loss + l1_lambda * l1_norm

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Evaluating the model on the validation fold
            accuracies = []
            f1s = []
            losses = []
            accuracy = 0
            count = 0
            loss = 0
            true_lables = []
            preds = []
            with torch.no_grad():
                for x, y in val_loader:
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
            accuracies.append(mean_accuracy)
            mean_loss = loss / count
            losses.append(mean_loss)
            true_lables = torch.Tensor(true_lables)
            preds = torch.Tensor(preds)
            f1 = f1_score(true_lables, preds, average='macro')
            f1s.append(f1)

        return torch.Tensor.mean(torch.Tensor(losses)), torch.Tensor.mean(torch.Tensor(accuracies)), torch.Tensor.mean(torch.Tensor(f1s))

    def validate(self, train_examples, X, Y, test_examples):
        normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG-16 - ImageNet Normalization

        # training data transformations
        train_transform = T.Compose([
            T.Resize(256),
            T.ColorJitter(),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            normalize
        ])

        train_loader = DataLoader(
                dataset=ShotDataset(train_examples, train_transform), 
                shuffle=True
            )

        # test data transformations
        eval_transform = T.Compose([
            T.Resize(256),
            T.ToTensor(),
            normalize
        ])

        test_loader = DataLoader(
            dataset=ShotDataset(test_examples, eval_transform),
            shuffle=False
        )

        # re-training the model on the whole training data
        self.model.train()
        for data in train_loader:
            X, Y = data
            X = X.to(self.device)
            Y = Y.to(self.device)
            output = self.model(X)
            loss = self.criterion(output, Y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # evaluating the model on test data
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        true_lables = []
        preds = []
        with torch.no_grad():
            for x, y in test_loader:
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
        precision = precision_score(true_lables, preds, average='macro')
        recall = recall_score(true_lables, preds, average='macro')
        cm = confusion_matrix(true_lables, preds)
        self.model.train()
        return mean_accuracy, mean_loss, f1, precision, recall, cm