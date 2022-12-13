import torch
import torch.nn as nn
from torch.autograd import Variable
from torchmetrics.functional.classification import binary_auroc, binary_accuracy

class classifiers(nn.Module):
    def __init__(self,  threshold = 0.5):
        super(classifiers, self).__init__()
        self.threshold = threshold
        self.classifier = nn.Linear(3, 1) 
        self.relu = nn.ReLU()
        
    def VotingClassifier(self, inputs):
        Voting = torch.empty(inputs.size()[1])
        Votes = torch.mode(inputs > self.threshold, dim = 0 ).values
        for i in range(inputs.size()[1]):
            vote = torch.mode(inputs > self.threshold, dim = 0 ).values[i]
            if vote:
                Voting[i] = inputs[:, i][inputs[:, i] > self.threshold].mean()
            else:
                Voting[i] = inputs[:, i][inputs[:, i] <= self.threshold].mean()
        return Voting
    
    def AverageClassifier(self, inputs):
        return inputs.mean(dim = 0)
    
    def Linear(self, x):
        x = self.classifier(x)
        return x