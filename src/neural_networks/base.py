import torch
import abc

class NNBase(torch.nn.Module):

    def __init__(self, cuda):
        super(NNBase, self).__init__()

        if cuda:
            self._device = torch.device('cuda:0')
        else:
            self._device = torch.device('cpu')
            
    @property
    def device(self):
        return self._device
    
    @abc.abstractmethod
    def forward(self,input):
        pass

    @abc.abstractmethod
    def predict(self,predict):
        pass