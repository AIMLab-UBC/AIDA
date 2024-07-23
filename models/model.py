import timm
import torch
import torch.nn as nn
import torchvision.models as models
from models.functions import ReverseLayerF

class AIDA(nn.Module):

    def __init__(self, cfg):
        super(AIDA, self).__init__()
        
        resnet = models.resnet18(pretrained=True)
      
        modules = list(resnet.children())[:-3]
        self.feat = nn.Sequential(*modules)

        modules = list(resnet.children())[-3:-1]
        self.class_classifier_1 = nn.Sequential(*modules)

        self.class_classifier_2 = nn.Sequential()
        self.class_classifier_2.add_module('last_fc', nn.Linear(512, cfg['num_classes']))
        self.class_classifier_2.add_module('res_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(256 * 32 *32 , 256))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(256))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(256 , 100))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        

    def forward(self, input_data, alpha):
        
        input_data = input_data.expand(input_data.data.shape[0], input_data.data.shape[1], 
                                       input_data.data.shape[2], input_data.data.shape[3])


        feature = self.feat(input_data)
        feat_C = self.class_classifier_1(feature)
        feat_C = feat_C.view(-1, feat_C.shape[1]* feat_C.shape[2] * feat_C.shape[3])
        class_output = self.class_classifier_2(feat_C)
        
        feat_D = ReverseLayerF.apply(feature, alpha)
        feat_D = feat_D.view(-1, feat_D.shape[1] * feat_D.shape[2] * feat_D.shape[3])
                
        domain_output = self.domain_classifier(feat_D)
        
        return class_output, domain_output

    def forward_last_fc(self, input_data, alpha):
        
        input_data = input_data.expand(input_data.data.shape[0], input_data.data.shape[1], 
                                       input_data.data.shape[2], input_data.data.shape[3])


        feature = self.feat(input_data)
        feat_C = self.class_classifier_1(feature)
        
        feat_C = feat_C.view(-1, feat_C.shape[1]* feat_C.shape[2] * feat_C.shape[3])
        class_output = self.class_classifier_2(feat_C)
        
        feat_D = ReverseLayerF.apply(feature, alpha)
        feat_D = feat_D.view(-1, feat_D.shape[1] * feat_D.shape[2] * feat_D.shape[3])
                
        domain_output = self.domain_classifier(feat_D)
        
        return class_output, domain_output


