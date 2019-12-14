'''
Written by Saurav Rai
For picking online pairs for training on casia dataset
using file casialist.txt: contains list of all images in casia train dataset

'''

import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn import metrics
import torch
from collections import defaultdict
import glob
import random

class CasiaFaceDataset(Dataset):
    def __init__(self, noofpairs = 4, transform = None ,is_train = True, trainfile = None):
        super().__init__()
        
        self.transform = transform
        self.istrain = is_train
        self.imagelist = self.get_id_imagelist()
        self.noofcategories = len(self.imagelist) - 1
        #print('noofcategories', self.noofcategories)

        self.noofpairs = noofpairs
        self.train_list = self.create_pairs()

    def get_id_imagelist(self):
        f = open('casialist.txt')
        lines = f.readlines()
        #print(len(lines),lines[0:5])
        subjectdict = dict()
        for name in lines[:]:
            subject = name.split('/')[0]
            #if subject not in subjectdict:
            subjectdict.setdefault(subject,[])
            subjectdict[subject].append(name)
        f.close()
        imagelist = []
        for i,(key,value) in enumerate(subjectdict.items()):
            #print(i,key,len(value))
            imagelist.append((i,key,value))
        return imagelist

    def get_random_two_images(self, tupleA, tupleB):
            classA= tupleA[0]
            classB = tupleB[0]
            listA = tupleA[2]
            listB = tupleB[2]
            imageA = np.random.choice(listA)
            imageB = np.random.choice(listB)
            #if classA!= classB:  Add this later
            while(imageA==imageB):
                imageB = np.random.choice(listB)

            return "/".join(imageA.split("/")[-2:]), classA, "/".join(imageB.split("/")[-2:]),classB

    def create_pairs(self):
            # returns 2D list of the format "[pathToImageA, pathToImageB, label]"
            pairsList = []  # empty list for all the information
            # iterate through all the categories in the dataset:
            for n in range(self.noofpairs):
                    #print ("Class", i+1,"out of", self.noOfCategories)
                        posCatList = np.arange(0,self.noofcategories)
                        i = np.random.choice(posCatList)
                        negativeCategoriesList = np.delete(np.arange(0,self.noofcategories),i)
                        # get "noOfPositiveExamples" of pairs with label "1":
                        imageA,c1, imageB,c2 = self.get_random_two_images(self.imagelist[i], self.imagelist[i])
                        #print('imageA ,c1 ,imageB,c2:',imageA,c1,imageB,c2)

                        pairsList.append([imageA,c1,imageB,c2,"1"])
                        # get "noOfNegativeExamples" of pairs with label "0":

                        j = np.random.choice(negativeCategoriesList)
                        imageA,c1, imageB,c2 = self.get_random_two_images(self.imagelist[i], self.imagelist[j])
                        #print('imageA ,c1 ,imageB,c2:',imageA,c1,imageB,c2)
                        pairsList.append([imageA,c1,imageB,c2,"0"])
            random.shuffle(pairsList)
            #print('The pairsList are:',pairsList)
            return pairsList
 

        
               
    def __len__(self):

        if self.istrain is True:
            #print(len(self.train_list))
            return len(self.train_list)
        '''
        else:
            return len(self.val_list)
        '''

    def __getitem__(self, i):#HERE i is the index / unique image_name
        
        # THIS IS THE TRAINING PART
        if self.istrain is True:
            image_name1 = self.train_list[i][0][:-1]
            image_name2 = self.train_list[i][2][:-1]
            id1 = self.train_list[i][1]
            id2 = self.train_list[i][3]
            label = self.train_list[i][4]
           
            path_img1 = os.path.join('/data/Saurav/DB/CASIAaligned/',image_name1) #Location to the image 
            path_img2 = os.path.join('/data/Saurav/DB/CASIAaligned/',image_name2)
            #print(path_img1,path_img2,id1,id2,label)
            if os.path.exists(path_img1) and os.path.exists(path_img2):
                #print('Both images exist')
                img1 = Image.open(path_img1).convert('L')
                img2 = Image.open(path_img2).convert('L')
                 	
                if self.transform:
                    img1 = self.transform(img1)
                    img2 = self.transform(img2)
                
                #print('Type image:',type(img1)) 
                #print('image name1 ,image name2 ,id1,id2,label:',img1,img2,id1,id2,label)
                return img1 , img2 , int(id1) ,int(id2) ,int(label)
        '''
        else:
            image_name1 = self.val_list[i][0]
            image_name2 = self.val_list[i][1]
            id1 = self.val_list[i][2]
            id2 = self.val_list[i][3]
            label = self.val_list[i][4]
            #print('image name1 ,image name2 ,id1,id2,label:',image_name1,image_name2,id1,id2,label)
            path_img1 = os.path.join('/data/Saurav/DB/vggface2/aligned_train_256_jpg/',image_name1)
            path_img2 = os.path.join('/data/Saurav/DB/vggface2/aligned_train_256_jpg/',image_name2)
            #path_img1 = os.path.join('/home/titanx/DB/vggface2/aligned_train_256_jpg/',image_name1)
            #path_img2 = os.path.join('/home/titanx/DB/vggface2/aligned_train_256_jpg/',image_name2)
            if os.path.exists(path_img1) and os.path.exists(path_img2):
                img1 = Image.open(path_img1).convert('L')
                img2 = Image.open(path_img2).convert('L')
                 	
                if self.transform:
                    img1 = self.transform(img1)
                    img2 = self.transform(img2)
                
                return img1, img2, int(label)
                #print('Type image:',type(img1)) 
                #print('image name1 ,image name2 ,id1,id2,label:',img1,img2,id1,id2,label)
          '''
'''
#For checking the Working of the the Datalodaer class 
casia =CasiaFaceDataset()
casia[2]
'''
