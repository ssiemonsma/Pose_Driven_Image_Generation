"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
# from options.train_options import TrainOptions
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.gan_trainer import GAN_Trainer
import os
import torch
from dataset_generator import Dataset_Generator_Aisin
from NADS_Net_model import NADS_Net

torch.set_default_tensor_type(torch.FloatTensor)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4,5"


# Holds Training options
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Training Options
opt = Namespace()
opt.gpu_ids = [0]
opt.ngf = 64
opt.crop_size = 256
opt.display_winsize = 256
opt.semantic_nc = 26
opt.num_D = 2
opt.output_nc = 3
opt.contain_dontcare_label = False
opt.save_latest_freq = 500
opt.print_freq = 1
opt.display_freq = 1
opt.continue_train = True
opt.lr = 0
opt.D_steps_per_G = 1
opt.aspect_ratio = 1.0
opt.batchSize = 1
opt.beta1 = 0.0
opt.beta2 = 0.9
opt.cache_filelist_read =True
opt.cache_filelist_write=True
opt.checkpoints_dir='./checkpoints'
opt.coco_no_portraits=False
opt.crop_size=256
opt.dataroot='./datasets/cityscapes/'
opt.dataset_mode='coco'
opt.debug=False
opt.display_winsize=256
opt.gan_mode='hinge'
opt.init_type='xavier'
opt.init_variance=0.02
opt.isTrain=True
opt.label_nc=0
opt.lambda_feat=10.0
opt.lambda_kld=0.05
opt.lambda_vgg=10.0
opt.load_from_opt_file=False
opt.load_size=286
opt.max_dataset_size=9223372036854775807
opt.model='pix2pix'
opt.nThreads=0
opt.n_layers_D=4
opt.name='NADS_Net_dataset'
opt.ndf=64
opt.nef=16
opt.netD='multiscale'
opt.netD_subarch='n_layer'
opt.netG='SPADE'
opt.ngf=64
opt.niter=50
opt.niter_decay=0
opt.no_TTUR=False
opt.no_flip=False
opt.no_ganFeat_loss=False
opt.no_html=False
opt.no_instance=True
opt.no_pairing_check=False
opt.no_vgg_loss=False
opt.norm_D='spectralinstance'
opt.norm_E='spectralinstance'
opt.norm_G='spectralspadesyncbatch3x3'
opt.num_D=2
opt.num_upsampling_layers='normal'
opt.optimizer='adam'
opt.output_nc=3
opt.phase='train'
opt.preprocess_mode='resize_and_crop'
opt.serial_batches=False
opt.tf_log=False
opt.which_epoch='latest'
opt.z_dim=256

# load the dataset
training_JSON_path = '/localscratch/Users/ssiemons/NADS-Net/Aisin_Dataset/Train_keypoint_Annotations.json'
raw_images_path = '/localscratch/Users/ssiemons/NADS-Net/Aisin_Dataset/Raw_Images/'
seatbelt_masks_path = '/localscratch/Users/ssiemons/NADS-Net/Aisin_Dataset/Seatbelt_Mask/'
num_dataloader_threads = 10
train_dataset = Dataset_Generator_Aisin(training_JSON_path, raw_images_path, seatbelt_masks_path, False, augment=False)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, num_workers=num_dataloader_threads, pin_memory=True, shuffle=True, drop_last=False)
dataset_size = len(train_dataset)
print('Number of training samples = %i' % dataset_size)

# Initialize NADS-Net (used for producing losses to help train Generator)
NADS_Net = NADS_Net(True, True, False).to(torch.device("cuda:1"))
NADS_Net.load_state_dict(torch.load('weights_training_with_tiny_PVT.pth',  map_location='cuda:1'))

# Create a trainer for the GAN
trainer = GAN_Trainer(opt, NADS_Net)

# Create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# This is used to print training statistics and errors
visualizer = Visualizer(opt)

# Training Loop
for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data in enumerate(dataloader, start=iter_counter.epoch_iter):
        real_images, keypoint_heatmaps, PAFs, seatbelt_segmentations, keypoint_heatmaps_small, PAFs_small = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].float().cuda(), data[4].float().cuda(), data[5].float().cuda()

        data = {}
        data['input_semantics'] = torch.cat((keypoint_heatmaps, PAFs, seatbelt_segmentations), dim=1).type(torch.FloatTensor)
        data['image'] = real_images

        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data)

        # train discriminator
        trainer.run_discriminator_one_step(data)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            maxed_input = torch.max(data['input_semantics'], dim=1).values
            visuals = OrderedDict([('input_label', maxed_input),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
