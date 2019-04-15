#Written by Saurav Rai on 25th Dec 2018
#Implementation of Face Recognition with Contrastive Convolution, Chunrui Han :ECCV2018
#Contrastive 4 network, trained on CASIA webface dataset and tested on lfw dataset
# Results are written to LFW_performance.txt

#For running the software in GPU the command is: python main.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pdb
import os

from onlineCasiadataset_loader import CasiaFaceDataset  #for load training pairs
import numpy as np

from light_cnn import LightCNN_4Layers, LightCNN_9Layers
from contrastive_cnn import Contrastive_4Layers, Contrastive_10Layers, Contrastive_14Layers

from LFWDataset import LFWDataset
from eval_metrics import evaluate, myevaluate
from tqdm import tqdm
from logger import Logger
from PIL import *
from torch.autograd import Function

def extractpatches( x, patch_size): #Written by Saurav to eliminate nested for loop
    patches = x.unfold( 2, patch_size ,  1).unfold(3,patch_size,1) 
    bs,c,pi,pj, _, _  = patches.size()

    #EXTRACTING PATCHES FROM THE IMAGE Here l and f are the list of tensor patches
    l = [patches[:,:,int(i/pi),i%pi,:,:] for i in range(pi * pi)]
    f = [l[i].contiguous().view(-1,c*patch_size*patch_size) for i in range(pi * pi)] 

    #CONCATENATE THE 9 TENSOR PATCHES OF THE LIST INTO ONE 
    stack_tensor = torch.stack(f)
    #Change the order of the dimension
    stack_tensor = stack_tensor.permute(1,0,2)
    return stack_tensor


class GenModel(nn.Module):
   def __init__(self,feature_size ):
       super(GenModel,self).__init__() 
       self.f_size = feature_size  #512 for author's and 192 for lightcnn 
        #kernel generators using single fc layers
       self.g1 = nn.Linear(self.f_size*3*3, self.f_size*3*3)
       self.g2 = nn.Linear(self.f_size*2*2, self.f_size*3*3)
       self.g3 = nn.Linear(self.f_size*1*1, self.f_size*3*3)
       self.relu = nn.ReLU()
       #self.relu = nn.Sigmoid()
         
       self.conv3x3 = nn.Conv2d(self.f_size,self.f_size,3)

   def forward(self, x):
       # n is feature from basemodel of size bsx 192x5x5
       #kernels = torch.tensor([0],dtype=torch.float32,requires_grad=True).to('cuda')  
       kernel1 = torch.tensor([0],dtype=torch.float32,requires_grad=True).to('cuda')  
       bs, _,_,_= x.size()
       S0 = x
       p1 = extractpatches(S0,3)
       S1 = self.relu(self.conv3x3(S0))
       p2 = extractpatches(S1,2)
       S2 = self.relu(self.conv3x3(S1))
       p3 = extractpatches(S2,1)
       kk1 = self.relu(self.g1( p1)) #Applies g1 on p1 to 3x3 patches to get kernels
       kk2 = self.relu(self.g2( p2)) 
       kk3 = self.relu(self.g3( p3)) 
       #kernels =  torch.cat((k1, k2,k3), dim = 0).transpose(1,0) #bsx14x1728
       kernels1 =  torch.cat((kk1, kk2,kk3), dim = 1)#.transpose(1,0) #bsx14x1728
       return kernels1

'''
    W : Regressor for pairwise similarity for verification on o/p of contrastive convolution
'''
class Regressor(nn.Module):
    def __init__(self, n):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(n, 1)
        self.relu = nn.ReLU() 
    def forward(self, x):
        bs, c = x.size()  #128 * 686
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x

'''
    Identity Regressor for the kernel - H(K)
'''
class Identity_Regressor(nn.Module):
    def __init__(self, n, classes):
        super(Identity_Regressor, self).__init__()
        self.fc1 = nn.Linear(n, 256) #here n = 14*512*3*3
        #self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

    def forward(self, x):
        bs, m, n = x.size()  # m no of kernel and n size of each kernel
        x = x.view(-1,n*m)   
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

def  train(args, basemodel, idreg_model, genmodel, reg_model, device, train_loader, optimizer, criterion,criterion1, iteration):

    genmodel.train()
    reg_model.train()
    idreg_model.train()
     
    for batch_idx ,(data_1, data_2, c1, c2, target) in enumerate(train_loader):
        data_1, data_2, c1, c2, target = (data_1).to(device), (data_2).to(device), torch.from_numpy(np.asarray(c1)).to(device), torch.from_numpy(np.asarray(c2)).to(device), torch.from_numpy(np.asarray(target)).to(device)
        
        target = target.float().unsqueeze(1)

        optimizer.zero_grad()

        A_list, B_list, org_kernel_1, org_kernel_2 = compute_contrastive_features(data_1, data_2, basemodel, genmodel, device)
        reg_1 = reg_model(A_list)
        reg_2 = reg_model(B_list)
        SAB = (reg_1 + reg_2)/2.0

        loss1 =  criterion1(SAB, target)  #pairwise similarity loss
        
        hk1 = idreg_model(org_kernel_1)
        hk2 = idreg_model(org_kernel_2)

        loss2 = 0.5 * ( criterion(hk1, c1) + criterion(hk2, c2))  #identity kernel loss for each image in pair
        loss  = loss2 + loss1

        loss.backward()
        #pdb.set_trace() 
        '''
        nn.utils.clip_grad_value_( list(genmodel.parameters()) + list( reg_model.parameters()) + list(idreg_model.parameters()), clip_value = 5)
        '''
        optimizer.step()
        #if iteration % args.log_interval == 0:
        print('Train iter: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} {:.4f} {:.4f}'.format(
             iteration, batch_idx * len(data_1), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item(), loss1.item(),loss2.item()))


def compute_contrastive_features(data_1,data_2,basemodel, genmodel, device):
    #pdb.set_trace()
    data_1, data_2 = (data_1).to(device), (data_2).to(device)

    data_1 = basemodel(data_1)
    data_2 = basemodel(data_2)
    #print(data_1.size())
    kernel_1 = genmodel(data_1).to(device)
    kernel_2 = genmodel(data_2).to(device) #B x 14 x 1728
    
    norm_kernel1 = torch.norm(kernel_1,2,2)
    norm_kernel2 = torch.norm(kernel_2,2,2)
    norm_kernel1_1 = torch.unsqueeze(norm_kernel1,2)
    norm_kernel2_2 = torch.unsqueeze(norm_kernel2,2)
    kernel_1 = kernel_1 / norm_kernel1_1
    kernel_2 = kernel_2 / norm_kernel2_2
      
    F1, F2 = data_1, data_2

    Kab = torch.abs(kernel_1 - kernel_2)  # B x 14 x 1728
    
    
    bs, featuresdim, h, w = F1.size()
    F1 = F1.view(1,bs*featuresdim,h,w)
    F2 = F2.view(1,bs*featuresdim,h,w)
    noofkernels = 14
    kernelsize = 3
    T = Kab.view(noofkernels* bs, -1, kernelsize,kernelsize ) 
    
    F1_T_out = F.conv2d(F1, T, stride = 1, padding = 2, groups = bs)
    F2_T_out = F.conv2d(F2, T, stride = 1, padding = 2, groups = bs)
    p,q,r,s = F1_T_out.size()
    A_list = F1_T_out.view(bs, -1)
    B_list = F2_T_out.view(bs, -1)

    return A_list, B_list, kernel_1, kernel_2


def adjust_learning_rate(optimizer, epoch):
    """decays learning rate very epoch at 100k iteration=40 epochs and 160 iterations=64 epochs"""

    for param_group in optimizer.param_groups:
        if epoch > 100000 and epoch < 140000:
            print('Learning rate is 0.01')
            param_group['lr'] = 0.01
        elif epoch > 140000:
            print('Learning rate is 0.001')
            param_group['lr'] = 0.001

def main():
    # settings
    parser = argparse.ArgumentParser(description='PyTorch Contrastive Convolution for FR')
    parser.add_argument('--batch_size', type=int, default = 64 , metavar='N',
                        help='input batch size for training (default: 64)')
    
    parser.add_argument('--epochs', type=int, default = 80, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--iters', type=int, default = 200000, metavar='N',
                        help='number of iterations to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
  
    parser.add_argument('--pretrained', default = False, type = bool,
                       metavar='N', help='use pretrained ligthcnn model:True / False no pretrainedmodel )')

    parser.add_argument('--basemodel', default='ContrastiveCNN-4', type=str, metavar='BaseModel',
                       help='model type:ContrastiveCNN-4 LightCNN-4 LightCNN-9, LightCNN-29, LightCNN-29v2')


    parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                       help='path to save checkpoint (default: none)')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                       help='path to latest checkpoint (default: none)')

    parser.add_argument('--start-epoch', default = 0, type=int, metavar='N',
                       help='manual epoch number (useful on restarts)')




    #Testing on LFW settings 
    parser.add_argument('--lfw-dir', type=str, default='/data2/Saurav/DB/lfw_mtcnnpy_256/', //path of the LFW dataset 
                    help='path to dataset')
    parser.add_argument('--lfw_pairs_path', type=str, default='lfw_pairs.txt',
                    help='path to pairs file')
    parser.add_argument('--test_batch_size', type=int, default = 128, metavar='BST',
                    help='input batch size for testing (default: 1000)')
    parser.add_argument('--compute_contrastive', default = True, type = bool,
                     metavar='N', help='use contrastive featurs or base mode features: True / False )')


    parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')

    #Training dataset on Casia

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
    parser.add_argument('--root_path', default='/data/Saurav/DB/CASIAaligned/', type=str, metavar='PATH', //path of the CASIA dataset
                    help='path to root path of images (default: none)')

    parser.add_argument('--num_classes', default=10574, type=int,
                    metavar='N', help='number of classes (default: 10574)')
    
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    #torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
       
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(128),
        transforms.ToTensor() ])
     
    test_loader = torch.utils.data.DataLoader(LFWDataset(dir=args.lfw_dir,pairs_path=args.lfw_pairs_path,
                                           transform=test_transform),  batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    transform=transforms.Compose([
                transforms.Resize(128),  #Added only for vggface2 as images are of size 256x256
                #transforms.CenterCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),])
     
    

    if args.basemodel == 'ContrastiveCNN-4':
        basemodel = Contrastive_4Layers(num_classes=args.num_classes)
        print('4 layer model')
    else:
        print('Model not found so existing.')
        assert(False)
    
    if args.pretrained is True:
 
        print('Loading pretrained model')

        pre_trained_dict = torch.load('./LightenedCNN_4_torch.pth', map_location = lambda storage, loc: storage) 
 
        model_dict = basemodel.state_dict()
        basemodel = basemodel.to(device)  #lightcnn model
        #only for ligthcnn4
        pre_trained_dict['features.0.filter.weight'] = pre_trained_dict.pop('0.weight')
        pre_trained_dict['features.0.filter.bias'] = pre_trained_dict.pop('0.bias')
        pre_trained_dict['features.2.filter.weight'] = pre_trained_dict.pop('2.weight')
        pre_trained_dict['features.2.filter.bias'] = pre_trained_dict.pop('2.bias')
        pre_trained_dict['features.4.filter.weight'] = pre_trained_dict.pop('4.weight')
        pre_trained_dict['features.4.filter.bias'] = pre_trained_dict.pop('4.bias')
        pre_trained_dict['features.6.filter.weight'] = pre_trained_dict.pop('6.weight')
        pre_trained_dict['features.6.filter.bias'] = pre_trained_dict.pop('6.bias')
        pre_trained_dict['fc1.filter.weight'] = pre_trained_dict.pop('9.1.weight')
        pre_trained_dict['fc1.filter.bias'] = pre_trained_dict.pop('9.1.bias')
        pre_trained_dict['fc2.weight'] = pre_trained_dict.pop('12.1.weight')
        pre_trained_dict['fc2.bias'] = pre_trained_dict.pop('12.1.bias')
        my_dict = {k: v for k, v in pre_trained_dict.items() if ("fc2" not in k )}  
        model_dict.update(my_dict) 
        

        basemodel.load_state_dict(model_dict, strict = False)             


    basemodel = basemodel.to(device)    
   
    
    genmodel = GenModel(512).to(device)       #kernel generator
    
    reg_model = Regressor(686).to(device)   #contrastive convolution o/p for binary regression

    idreg_model = Identity_Regressor(14 * 512 * 3 * 3, args.num_classes).to(device)  #Kernel o/p for Identity recongition
    params =  list(basemodel.parameters()) +  list(genmodel.parameters()) + list(reg_model.parameters()) + list(idreg_model.parameters())

    optimizer = optim.SGD(params , lr=args.lr, momentum=args.momentum)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['iterno']
            genmodel.load_state_dict(checkpoint['state_dict1'])
            basemodel.load_state_dict(checkpoint['state_dict2'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Test acc at checkpoint was:',checkpoint['testacc'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
   
    criterion2   = nn.CrossEntropyLoss().to(device)         #for kernel loss: Identification loss
    criterion1   = nn.BCELoss().to(device)         #for Similarity loss
    
    print('Device being used is :' + str(device))

    for iterno in range(args.start_epoch + 1, args.iters + 1): 
        
        adjust_learning_rate(optimizer, iterno)
        
        traindataset = CasiaFaceDataset(noofpairs = args.batch_size, transform=transform,is_train = True)

        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        #train(args, basemodel, idreg_model, genmodel, reg_model, device, train_loader, optimizer, criterion2, criterion1,iterno)
        #if iterno > 0 and iterno%1000==0:
        testacc =  test( test_loader, basemodel, genmodel, reg_model,  iterno, device, args)
        f = open('LFW_performance.txt','a')
        f.write('\n'+str(iterno)+': '+str( testacc*100));
        f.close() 
        print('Test accuracy: {:.4f}'.format(testacc*100))
            
        if iterno > 0 and iterno%10000==0:
            save_name = args.save_path +'base_gen_model' + str(iterno) + '_checkpoint.pth.tar'
            save_checkpoint({'iterno': iterno ,   'state_dict1': genmodel.state_dict(),'state_dict2':basemodel.state_dict(),
               'optimizer': optimizer.state_dict(),
               'testacc':testacc}, save_name)
            
def save_checkpoint(state, filename):
    torch.save(state, filename)


def test(test_loader, basemodel, genmodel, reg_model, epoch, device, args):
    # switch to evaluate mode
    basemodel.eval()

    labels, distance , distances = [], [],[]

    pbar = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for batch_idx, (data_a, data_b, label) in pbar:
            data_a, data_b = data_a.to(device), data_b.to(device)
            # compute  feature after computing contrastive kernel
             
            if args.compute_contrastive:
                out1_a, out1_b, k1, k2 = compute_contrastive_features(data_a,data_b,basemodel, genmodel, device)
                
               
                SA = reg_model(out1_a)
                SB = reg_model(out1_b)
                SAB = (SA + SB) / 2.0
            SAB = torch.squeeze(SAB,1)
            
            #print('SAB :',SAB)
            distances.append(SAB.data.cpu().numpy())
            #print('distances len',len(distances))
            labels.append(label.data.cpu().numpy())
            #print('labels len',len(labels))

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx * len(data_a), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader)))
        
        #print('target ',labels)
        #print('distances ',distances)

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances  =  np.array([subdist for dist in distances for subdist in dist]) 
        print('target length',len(labels))
        print('distances length ',len(distances))

        accuracy = evaluate(1-distances,labels)
        return np.mean(accuracy)
      
if __name__ == '__main__':
    main()
