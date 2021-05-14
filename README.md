# Pose-Driven Image Generation

## Overview
[NADS-Net](https://arxiv.org/abs/1910.03695) was a first-of-its-kind neural network that extracts both poses and seatbelt segmentations from in-vehicle video frames.  It was trained on a proprietary dataset.  Unfortunately, the training dataset is not very large nor diverse, so the network experiences overfitting issue.  To address this issue, this project explored the possibility of using the generator from a GAN to augment the poses of the people present in the training dataset.  If successful, this could serve as a very useful form of data augmentation.

For this project, NADS-Net was modified to have a [Pyramid Visision Transformer (PVT) backbone](https://arxiv.org/abs/2102.12122) in order to make the network smaller and more efficient (due to concerns about GPU memory constaints).

A [SPADE](https://arxiv.org/abs/1903.07291) generator was selected with a multi-scale patch-gan discriminator.

The entire project is implemented in PyTorch.

## Installation of Dependencies
(Note: I still need to update requirements.txt before you do this!!!  This will be completed and validated on a fresh virtual environment before the deadline.)  
This project was validated with Python 3.8.8 on Linux.  All of the project dependencies are located requirements.txt and can be installed with:
> pip install -r requirements.txt

## Hardware Requirements
This project was developed a server with multiple GPUs available.  The training would not fit on a single 11 GB GPU, so the work was divided between two 1080 Ti GPUs, one hosting the GAN and the other hosting NADS-Net.  The code could be easily modified if a single high-VRAM GPU is available, but the code was written to run on 2 GPUs simultaneously.

## Dataset and Pretrained Weights
Unfortunately, since the NADS-Net dataset is proprietary, I cannot share it.  However, I have opted to include a small sample of this dataset (one training sample and one validation sample) so that the code will run.  In addition, pretrained weights for the generator, discriminator, and NADS-Net are also provided.
[**Download this folder**](https://iowa-my.sharepoint.com/:f:/g/personal/ssiemons_uiowa_edu/ErrJ0JpiEd1IhPPFgPk51HgBLgIX5qAyfgvOI0HlUlQweg?e=Y9gHxr) and place it at the same level as the GAN and NADS-NET_with_PVT folders.

The sample data includes the folders Raw_Images and Seatbelt_Mask, along with Train_keypoint_Annotation_sample.json and Valid_keypoint_Annotation_sample.json.  Raw_Images contains two frames from an in-vehicle camera (one for training and one for validation).  Seatbelt_Mask contains seatbelt segmentations for these same frames.  The JSON files contain manually annotated body keypoint coordinates.  For reference, I'll describe the order these keypoints are ordered in:
> 0: right wrist  
> 1: right elbow  
> 2: right shoulder  
> 3: neck  
> 4: left shoulder  
> 5: left elbow  
> 6: left wrist  
> 7: right hip  
> 8: left hip

In the code, these coordinates are used by dataset_generatory.py's Dataset_Generator_Aisin class to create individual heatmap layers for each of these keypoints, with a Gaussian distribution being used to softly mark each of these locations.

In addition, these keypoint coordinates are also used to create [Part Affinity Fields (PAF)](https://arxiv.org/abs/1611.08050).  Part Affinity Fields each have a channel depth of 2 since they are a series of two-dimensional vectors pointing from one keypoint to another.  In total, this project uses 8 different PAFs (for a channel depth of 16), listed below for reference.
> 0-1: right shoulder <-> neck  
> 2-3: right shoulder <-> right elbow  
> 4-5: right elbow <-> right wrist  
> 6-7: under head <-> left shoulder  
> 8-9: left shoulder <-> neck  
> 10-11: left elbow <-> left wrist  
> 12-13: neck <-> left hip  
> 14-15: neck <-> right hip

Overall, when using the GAN's generator, a concatenated input of size 26x384x384 is created, consisting of the 9 keypoint heatmaps, the 8 PAFs, and the seatbelt segmentation.  The output of that generator is 3x384x384 RGB image.

## Code Overview
The Code is split into two different folders: GAN and NADS-Net_with_PVT.  The GAN folder is where you can train the GAN, and the NADS-Net_with_PVT folder is where you can train the modified NADS-Net architecture.  Each of these can be trained simply by running train.py in their respective folders.  Each of the train.py files contain a variety of training parameters that you can modify, with more details be provided as code comments.

Both the GAN and NADS-Net_with_PVT folders contain the same file names and folders since they both use the same two networks: NADS-Net and the SPADE GAN.  However, there are some notable differences between the files.  The two train.py files are completely different and simply share a name.  The dataset_generator.py files are similar, but the one in NADS-Net_with_PVT has been modified to allow for limb pose augmentation.

### The GAN
In each of the two major folders, the "models" folder contains all of the models needed to create the GAN.  At its highest level, the GAN is defined in the GAN_Model class of model/gan_model.py.  This class contains both SPADE generator and the multi-scale patch-GAN discriminator, along with the logic needed pass data through these networks and compute losses.

The loss functions used to train the GAN used are largely present in GAN/models/networks/loss.py.  These include a GAN loss for the generator and discriminator (in the form of Hinge losses for this project), a VGG perceptual loss, and a feature-matching loss which uses intermediate layers of the discriminator and is another type of perceptual loss.  The feature-matching loss is not actually in loss.py, but instead directly computed in gan_model.py.

In addition, the generator is also trained with three losses derived from a NADS-Net inference on the fake images: a keypoint heatmap MSE loss, a Part Affinity Field (PAF) loss, and a seatbelt Dice-BCE loss.  These losses are all computed in gan_model.py, with the Dice-BCE loss defined in GAN/models/networks/loss.py.  Dice-BCE loss is simply a 1-to-1 linear combination of a dice loss and a binary cross entropy loss.

When training the GAN, you can weight these various losses by changing their corresponding lambda values in the training settings variables of GAN/train.py.  Most of the high-level training logic for the GAN is contained in GAN/trainers/gan_trainer.py, which includes the GAN_Trainer class.  This class notably includes the functions run_generator_one_step() and run_discriminator_one_step(), which each infer on training data and perform the appropriate backpropagation.  It also contains the function save(), which will save the model weights.  The trainer can also be configured to allow the learning rate to day after a certain number of iterations.

By default, GAN/train.py is configured with a fixed learning rate of 0.0002, using an Adam optimizer.  While running GAN/train.py, the various training losses will by printed out at a rate dictated by opt.print_freq, which indicates how many batches must pass before an update is printed to the console.  There are additional parameters opt.save_latest_freq and opt.save_epoch_freq, which dictate how often to save the GAN's weights.  The weights are saved in GAN/checkpoints/NADS_NET_dataset/.

In addition, the opt.display_freq dictates how often a set of images are saved for visualization.  These images include: a composite of the 26x384x384 tensor inputted into the generator, the fake generated image, and the real image.  These images are saved in GAN/checkpoints/NADS_NET_dataset/web/images/, and they can be viewed together by opening index.html in the directory above.


### NADS-Net with PVT Backbone
The PVT backbone of the modified NADS-Net architecure is found in the PVT folders.  This project used the "tiny PVT" for optimal efficiency, which is defined in the pvt_tiny() function in PVT/pvt.py.

The NADS-Net model is defined in NADS_Net class of NADS_NET_model.py.  Immediately following its PVT backbone, there is a feature pyramid network (FPN) defined in the class FPN.  Then, there are three different convolution heads for calculating the keypoint heatmaps, part affinity fields, and seatbelt segmentation.  The keypoint heatmaps and PAF heads are contructed using the Map_Branch class, and the seatbelt segmentation head is constructed using the Segmentation_Branch class.  Please note that NADS-Net output the keypoint heatmap and PAFs at a reduced 96x96 spatial resolution, whereas the seatbelt segmentation branch outputs a larger 384x384 image.

Training is done by running NADS-NET_with_PVT/train.py, which contains a variety of training parameters, including the starting learning rate and batch size.  This can can be configured to use either a fixed or a metric-based learning rate scheduler.  By default, the learning rate is set at 1e-5 and is set to decrease by a factor of 0.8 for every 4 epochs that pass without improving the total validation loss.  An SGD optimizer is used by default.

The main feature of NADS-NET_with_PVT/train.py is that it can augment the arm poses of half of the training data and then generate a fake image in this pose using the SPADE generator.  The arm keypoint coordinates can be either "slightly" or "highly" modified, as set by arm_augment_type.  This coordinate augmentation occurs directly in the __getitem__() method of the Dataset_Generator_Aisin class in NADS-NET_with_PVT/dataset_generator.py.  This generator creates the keypoint heatmaps, part affinity fields, and seatbelt segmentation, which are concatenated together in train.py and fed into the pretrained SPADE generator to create fake images in the augmented poses for 50% of the samples in each batch (Note: this had to occur outside of the dataset generator since GPU operations aren't allowed in a multi-threaded generator).  The images (50% fake and augmented, and 50% real and unaugmented) and their ground truth label maps are then augmented with standard random resizings and translations before being fed into NADS-Net.

NADS-Net is then trained with a linear combination of the keypoint heatmap MSE, the part affinity field MSE, and the seatbelt segmentation Dice-BCE loss.  While training, a tqdm progress bar is updated with the latest batch losses, and then the average epoch losses are printed after every full epoch.  The training loop is set to only save the NADS-Net weights when an epoch's validation loss is lower than any previous epoch.
