import torch.nn as nn
from src.utils import square_euclidean_metric
import torch
import torchvision.models as model
import torch.nn.functional as F

""" Optional conv block """
def conv_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Dropout(p = 0.2),
        nn.MaxPool2d(kernel_size= 2, stride= 2),
        
    )


""" Define your own model """
class FewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=12):
        super(FewShotModel,self).__init__()
        self.encoder1 = conv_block(x_dim,z_dim)
        self.encoder2 = conv_block(z_dim,x_dim)
    def forward(self, x):

        x = self.encoder1(x)
        x = self.encoder2(x)
        
        embedding_vector = x.view(x.size()[0], -1)
        
        return embedding_vector
    
    
class Prototypical(nn.Module):
    def __init__(self):
        super(Prototypical, self).__init__()
        self.shot = FewShotModel()
        self.query = FewShotModel()
        self.d = Dist()
    def forward(self, data_shot, data_query, args):
            
        h = data_shot.size(2)
        w = data_shot.size(3)
        shot_res = torch.reshape(data_shot, [args.nway, args.kshot, 3, h, w])
            
        support_set = []
        for i in range(args.nway):
            support_set.append(torch.squeeze(self.shot(shot_res[i, :, :, :, :]), 0))
                
        support_set = torch.stack(support_set)
        query_set = self.query(data_query)
            
        logits = self.d(support_set, query_set)
            
        return logits
              
class Dist(nn.Module):
    def __init__(self):
        super(Dist, self).__init__()
        
    def forward(self, shot, query):
        prototype = torch.mean(shot, 1) # size : (nway, output dimension of embedding)
        logits = square_euclidean_metric(query, prototype)
        
        return logits
    