# Product_Detection_Assignment (Infilect)

# Results
      
      mAP : 0.92
      precision: 0.99
      recall: 0.46

# Dependencies

Solution depends on the following main libraries:

      Tensorlfow
      Keras
      Tensorflow Object Detection API
      OpenCV

# Data Preparation
      
      Script Usage:
      
      Usage:
            python Data_Prep.py <Path_to_Product_Images_Parent_Folder> \
                                <Path_to_Shelves_Images_Folder> \
                                <Name_of_Output_Product_Images.csv> \
                                <Name_of_Output_Shelves_Images.csv>
                                
            eg: python Data_Prep.py GroceryDataset/ProductImagesFromShelves/ \
                                    GroceryDataset/ShelfImages/ \
                                    products_data.csv \
                                    shelf_data.csv

      ProductImagesFromShelves contains cut-out photos of goods from shelves in 11 subdirectories: 
      0 - not classified, 1 - Marlboro, 2 - Kent, etc. Files in the names contain information about the rack, 
      the position and size of the pack on it.
      
      There can be several photos of one rack. Accordingly, the same pack can fall into several pictures. Therefore, we
      need to break down not by pictures, and even more so not by packs, but by racks. This is necessary so that it does
      not happen that the same object, taken from different angles, ends up in both train and validation and make sure
      that in our breakdown there are enough representatives of each class both for training and for validation.
      
      NOTE: Some of the original photos were rotated so we have to rotate them to proper angle.(This can be done with
      imagemagik in terminal or any image viewer tools)
      
      Take a look at Data_Prep.py to find the corresponding code.
      
      Once CSV files are generated, we now need to make tf.records for training input using Making_TFrecords.py
      
# Data Augmentation And Making TF Records

      Usage:
            python Making_TFrecords.py <Path_to_shelf data CSV created using Data_Prep.py> \
                                <Path_to_product data CSV created using Data_Prep.py>
                                
            eg: python Making_TFrecords.py shelf_data.csv product_data.csv
      
      Random crop is done from the original given data which can be found in Making_TFrecords.py random_crop function.
      random_horizontal_flip etc augemntations are also which can be found in SSD config file.
      
# Detection Network
      
      MobileNet V1

# Parameters
      
      ALl the parameters used can be found in config file (Note: Only 1 Anchor Box per feature map is used)
      
# Running Training Proccess

Please, go to pack_detector/models/ssd_mobilenet_v1 and create two subfolders: eval and train. Now we have the pack_detector folder with the following content:

+data

      train.record
      eval.record
      pack.pbtxt
      
+models

      +ssd_mobilenet_v1
          +train
          +eval
          ssd_mobilenet_v1_pack.config
          
You can read the detailed description and purpose of these files here Running Locally (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md)

Do the following steps to run the process:

1) Install Tensorflow Object Detection API if you haven't done it before Installation
   (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

2) Go to the models/research/object_detection directory Download and untar pretrained model
        
        wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
   
3) Extract the file
        
        tar -xvzf ssd_mobilenet_v1_coco_2017_11_17.tar.gz

4) Copy your pack_detector folder to models/research/object_detection

5) Run train process

        python3 model_main.py --logtostderr \
          --train_dir=pack_detector/models/ssd_mobilenet_v1/train/ \
          --pipeline_config_path=pack_detector/models/ssd_mobilenet_v1/ssd_mobilenet_v1_pack.config
          
6) Run eval proccess

          python3 model_main.py
          --checkpoint_dir=pack_detector/models/ssd_mobilenet_v1/train \
          --pipeline_config_path=pack_detector/models/ssd_mobilenet_v1/ssd_mobilenet_v1_pack.config

7) Run tensorboard process

        tensorboard --logdir=pack_detector/models/ssd_mobilenet_v1

8) Generate .pb file from checkpont
         
         python3 export_inference_graph.py \
         --input_type image_tensor \
         --pipeline_config_path pack_detector/models/ssd_mobilenet_v1/ssd_mobilenet_v1_pack.config \
         --trained_checkpoint_prefix pack_detector/models/ssd_mobilenet_v1/train/model.ckpt-13756 \
         --output_directory pack_detector/models/ssd_mobilenet_v1/pack_detector_2018_06_03

# Inference 
      
      Usage:
            python Inference.py <path to pb file> \
                                <path to pack.pbtxt file> \
                                <path to shelf images> \
                                <path to Grocery Data parent folder> \
                                <path to shelf data CSV created>
        
        Eg : python Inference.py frozen_inference_graph_18628_1_anchor.pb \
                            pack_detector/data/pack.pbtxt \
                            ../GroceryDataset/ShelfImages \
                            ../GroceryDataset \
                            shelf_data.csv
                            
        Script runs on all test images from shelf_data.csv.Best Results are achived with mentioned .pb file and applying
        sliding window with non-max supression. The .pb file genererated from the checkpoint is not giving good results on
        full image but performing well on smaller part of image hence tried sliding window with non-max supression.

# Get Metrics 
      Usage:
            python Get_Metrics.py
            
            
# Q&A

1)What is the purpose of using multiple anchors per feature map cell?

      multiple anchors per feature map cell enables the network to predict multiple objects of different sizes per
      image location its a standard methodology
      
2)Does this problem require multiple anchors? Please justify your answer.

      It may not require mutiple anchors as size and shape of all products in our data are almost is the same range and depth
      of all images are almost in same range too, so 1 anchor box per cell can do the need in this use case and added to that
      lesser the no of anchor boxes faster the inference is and helps alot in deploying on mobile or low config devices
      
# Notes

      Inference.py code can be optimised more. Time of Inference.py can be reducing by running the 
      inference on images parallely with multiprocessing(currently its on for loop), Also I am saving Predictions of images
      in a text file to caliculate mAP and other metrics(this part of code can be optimised too) 
      
      Get_Metrics.py code be cleaned more.
      
      
