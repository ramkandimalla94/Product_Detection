# Product_Detection_Assignment

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

2) Go to the models/research/object_detection directory Download and untar pretrained model with:
   wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
   
3) tar -xvzf ssd_mobilenet_v1_coco_2017_11_17.tar.gz

4) Copy your pack_detector folder to models/research/object_detection

5) Run train process with: 
        python3 train.py --logtostderr \
          --train_dir=pack_detector/models/ssd_mobilenet_v1/train/ \
          --pipeline_config_path=pack_detector/models/ssd_mobilenet_v1/ssd_mobilenet_v1_pack.config
6) Run eval proccess on CPU with:
        CUDA_VISIBLE_DEVICES="" python3 eval.py \
          --logtostderr \
          --checkpoint_dir=pack_detector/models/ssd_mobilenet_v1/train \
          --pipeline_config_path=pack_detector/models/ssd_mobilenet_v1/ssd_mobilenet_v1_pack.config \
          --eval_dir=pack_detector/models/ssd_mobilenet_v1/eval

7) Run tensorboard process with:
        tensorboard --logdir=pack_detector/models/ssd_mobilenet_v1
