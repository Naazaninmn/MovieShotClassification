import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn, resnet18, resnet50, resnet101
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from model import ResNetModel


class Experiment:
    
    def __init__(self, opt):

        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        # self.model = vgg19_bn(pretrained=True)
        # self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=5)
        self.model = resnet101(pretrained=True)
        self.model.fc.out_features = 5
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True
        
        # for i in range(4):
        #     for param in self.model.features[i].parameters():
        #         param.requires_grad = False

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion = torch.nn.CrossEntropyLoss()


    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data):
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = self.criterion(logits, y)

        #L2 regularization
        # l2_lambda = 0.01
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
        loss = loss + l2_lambda * l2_norm

        #L1 regularization 
        # l1_lambda = 0.01
        # l1_lambda = 0.001
        # l1_norm = sum(abs(p).sum() for p in self.model.parameters())
        # loss = loss + l1_lambda * l1_norm

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

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
        precision = precision_score(true_lables, preds, average='macro')
        recall = recall_score(true_lables, preds, average='macro')
        cm = confusion_matrix(true_lables, preds)
        self.model.train()
        return mean_accuracy, mean_loss, f1, precision, recall, cm