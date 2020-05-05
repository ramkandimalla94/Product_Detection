"""
Usage:
python Data_Prep.py <Path_to_Product_Images_Parent_Folder> \
                    <Path_to_Shelves_Images_Folder> \
                    <Name_of_Output_Product_Images.csv> \
                    <Name_of_Output_Shelves_Images.csv>
eg: python Data_Prep.py GroceryDataset/ProductImagesFromShelves/ GroceryDataset/ShelfImages/ products_data.csv shelf_data.csv
"""

import cv2
import pandas as pd
from matplotlib import pyplot as plt
import os
import sys 
import glob 
import numpy as np
from sklearn.model_selection import train_test_split


'''
from PIL import Image
k=0
for i in prod_img:
    im = Image.open(i)
    width, height = im.size
    #print (width, height)
    k=k+float(height)
print(k/len(prod_img)) 
'''

def Prod_Data(Prod_Data_Path):

    """
    Tales Products data path folder as argument as gets the list of all files prsesent
    in the folder, as splits the string amd get the data acording to convention.
    """

    # Get list of all images
    prod_img =glob.glob(Prod_Data_Path+'*/*.png')

    p_images,category=[],[]
    x1,y1,x2,y2,w,h,shelf_name=[],[],[],[],[],[],[]

    for i in prod_img:

        i=i.replace('.png','')
        _1=i.split('/')[-1].split('.JPG_')
        (_1[0][:6])
        p_images.append(i.split('/')[-1].split('.JPG_')[0]+'.JPG')
        
        _2=_1[-1].split('_')
        x1.append(_2[0])
        y1.append(_2[1])
        w.append(_2[2])
        h.append(_2[3])
        x2.append(int(_2[0])+int(_2[2]))
        y2.append(int(_2[1])+int(_2[3]))
        category.append(i.split('/')[-2])
        shelf_name.append(_1[0][:6])

    #Making Data Frame to save in CSV
    product_data=pd.DataFrame({'image':p_images,'category':category,'shelf_name':shelf_name,
                                'x1':x1,'y1':y1,'x2':x2,'y2':y2,'w':w,'h':h}) 

    return product_data

def Shelf_Data(Shelf_Data_Path):

    """
    Tales Shelves data path folder as argument as gets the list of all files prsesent
    in the folder, as splits the string amd get the data acording to convention.
    """

    # Get list of all images
    shelf_img=glob.glob(Shelf_Data_Path+'*.JPG')

    s_images,shelf_name=[],[]
    for i in shelf_img:
        s_images.append(i.split('/')[-1])
        shelf_name.append(i.split('/')[-1][:6])

    #Making Data Frame to save in CSV    
    shelf_data=pd.DataFrame({'image':s_images,'shelf_name':shelf_name}) 

    return shelf_data

def main():

    try:
        
        product_data = Prod_Data(sys.argv[1])
        shelf_data = Shelf_Data(sys.argv[2])

        #Get list of unique Shelves from all the different shelves to split on this column
        shelves_all = list(set(shelf_data.shelf_name))

        #Test & Train split using Sklearn 
        shelves_train, shelves_test= train_test_split(shelves_all,test_size=0.25)

        def is_train(shelf_name): 
            return shelf_name in shelves_train
        
        # Makinf Train Flag in Data
        shelf_data['train'] = shelf_data.shelf_name.apply(is_train)
        product_data['train'] = product_data.shelf_name.apply(is_train)

        product_data.to_csv(sys.argv[3],index=False)
        shelf_data.to_csv(sys.argv[4],index=False)

        print ('CSV Files Created')

    except Exception as e:
        print (e)
    

if __name__== "__main__":
    main()
