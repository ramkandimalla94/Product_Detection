# Product_Detection_Assignment (Infilect)

# Dependencies

Solution depends on the following main libraries:

      Tensorlfow
      Keras
      Tensorflow Object Detection API
      OpenCV

# Data Preparation

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
      
# Data Augmentation
      
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

        CUDA_VISIBLE_DEVICES="" python3 eval.py \
          --logtostderr \
          --checkpoint_dir=pack_detector/models/ssd_mobilenet_v1/train \
          --pipeline_config_path=pack_detector/models/ssd_mobilenet_v1/ssd_mobilenet_v1_pack.config \
          --eval_dir=pack_detector/models/ssd_mobilenet_v1/eval
          
          python3 model_main.py
          --checkpoint_dir=pack_detector/models/ssd_mobilenet_v1/train \
          --pipeline_config_path=pack_detector/models/ssd_mobilenet_v1/ssd_mobilenet_v1_pack.config

7) Run tensorboard process

        tensorboard --logdir=pack_detector/models/ssd_mobilenet_v1


# Q&A

1)What is the purpose of using multiple anchors per feature map cell?

      multiple anchors per feature map cell enables the network to predict multiple objects of different sizes per
      image location.
      
2)Does this problem require multiple anchors? Please justify your answer.

      It may not require mutiple anchors
