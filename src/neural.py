import numpy as np
import torch
import torch.nn as nn

from . import constants as ck

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    """
    For other car belief.
    """
    def __init__(self, nx=ck.n_x_points, ny=ck.n_y_points, nc0=3,
        out=ck.n_x_actions*ck.n_y_actions,
        input_=6, # For our car.
        drop=0):
        super(CNN, self).__init__()

        # For our car
        nc1_ = 64
        self.drop1_ = nn.Dropout(drop)
        self.layer1_ = nn.Linear(input_, nc1_)
        self.nl1_ = nn.ReLU()

        self.drop2_ = nn.Dropout(drop)
        self.layer2_ = nn.Linear(nc1_, nx*ny*nc0)
        self.nl2_ = nn.ReLU()

        # For other cars
        nc1 = 32
        self.drop1 = nn.Dropout(drop)
        self.layer1 = nn.Conv2d(nc0*2, nc1, (5,3), stride=(2,2))
        nx = int((nx-3)/2)
        ny = int((ny-1)/2)
        # Output: x=200->98. y=5->2
        self.nl1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(nc1)

        nc2 = 64
        self.drop2 = nn.Dropout(drop)
        self.layer2 = nn.Conv2d(nc1, nc2, (5,1), stride=(2,1))
        nx = int((nx-3)/2)
        ny = ny
        # Output: x=98->47. y=2->2
        self.nl2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(nc2)

        nc3 = 128
        self.drop3 = nn.Dropout(drop)
        self.layer3 = nn.Conv2d(nc2, nc3, (5,2), stride=(2,1))
        nx = int((nx-3)/2)
        ny = ny-1
        # Output: x=47->22. y=2->1
        self.nl3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(nc3)

        nc4 = 128
        self.drop4 = nn.Dropout(drop)
        self.layer4 = nn.Linear(nx*ny*nc3, nc4)
        self.nl4 = nn.ReLU()

        self.drop5 = nn.Dropout(drop)
        self.layer5 = nn.Linear(nc4, out)
        self.nl5 = nn.Softmax(dim=-1)


        # Initialize weights according to the Xavier Glorot formula
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)
        # nn.init.xavier_uniform_(self.fc4.weight)

    # Forward computation. Backward computation is done implicitly.
    def forward(self, our_car, belief):
        # TODO: Batch.
        assert len(belief.size()) == 3
        belief = belief.view(1, belief.size()[0], belief.size()[1], belief.size()[2])
        
        our_car = torch.tensor([our_car.x, our_car.y, our_car.vx, our_car.vy,
            our_car.ax, our_car.ay]).view(1, 6)
        our_car = self.nl1_(self.layer1_(self.drop1_(our_car)))
        our_car = self.nl2_(self.layer2_(self.drop2_(our_car)))
        our_car = our_car.view(belief.size())

        inp = torch.cat([our_car, belief], axis=1)

        out = self.bn1(self.nl1(self.layer1(self.drop1(inp))))
        out = self.bn2(self.nl2(self.layer2(self.drop2(out))))
        out = self.bn3(self.nl3(self.layer3(self.drop3(out))))
        out = self.nl4(self.layer4(self.drop4(out.view(out.size()[0], -1))))
        out = self.nl5(self.layer5(self.drop5(out)))
        return out

