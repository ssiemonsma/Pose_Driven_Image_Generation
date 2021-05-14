from collections import OrderedDict
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.gan_trainer import GAN_Trainer
import os
import torch
from dataset_generator import Dataset_Generator_Aisin
from NADS_Net_model import NADS_Net
from util.get_gan_options import get_gan_options
from shutil import copyfile
import sys

torch.set_default_tensor_type(torch.FloatTensor)

# We need to use the first GPU for the generator and the second for NADS-Net
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# Get default training options
opt = get_gan_options()

# Customize notable variables
opt.continue_train = True   # Starting with latest_net_D.pth and latest_net_G.pth weights
opt.lr = 0.00002            # Learning rate
opt.batchSize = 1
opt.lambda_feat = 10        # Scales the feature-matching loss
opt.lambda_vgg = 10         # Scales the VGG loss
opt.lambda_keypoint = 100   # Scales the Keypoint heatmap MSE loss
opt.lambda_paf = 100        # Scales the Part affinity field MSE loss
opt.lambda_seatbelt = 1     # Scales the Seatbelt Dice-BCE loss
opt.gpu_ids = [0]           # Set GAN to run on first GPU of those visible
opt.netG='SPADE'            # Set generator to the SPADE generator
opt.num_D = 2               # We are using a multi-scale patch discriminator at 2 different scalings
opt.n_layers_D=4            # Number of layers in each discriminator
opt.semantic_nc = 26        # Concatenated input maps have a channel depth of 26
opt.output_nc = 3           # We are outputting an RGB image
opt.save_latest_freq = 500  # Save the latest weights every 500 batches
opt.save_epoch_freq = 10    # Save an extra copy of the weights every 10 epochs
opt.print_freq = 1          # For demonstration purposes, print losses after every batch
opt.display_freq = 1        # For demonstration purposes, save generated images every batch

# Copy over pretrained GAN weights, if continuing training
if opt.continue_train:
    if not os.path.exists('./checkpoints/NADS_Net_dataset/latest_net_G.pth'):
        if os.path.exists('../Data_and_Pretrained_Weights/Pretrained_Weights/latest_net_G.pth'):
            copyfile('../Data_and_Pretrained_Weights/Pretrained_Weights/latest_net_G.pth', './checkpoints/NADS_Net_dataset/latest_net_G.pth')
        else:
            sys.exit('Please download the folder Data_and_Pretrained_Weights from OneDrive first and place it at the same level as GAN and NADS-Net_with_PVT.')
    if not os.path.exists('./checkpoints/NADS_Net_dataset/latest_net_D.pth'):
        if os.path.exists('../Data_and_Pretrained_Weights/Pretrained_Weights/latest_net_D.pth'):
            copyfile('../Data_and_Pretrained_Weights/Pretrained_Weights/latest_net_D.pth', './checkpoints/NADS_Net_dataset/latest_net_D.pth')
        else:
            sys.exit('Please download the folder Data_and_Pretrained_Weights from OneDrive first and place it at the same level as GAN and NADS-Net_with_PVT.')

# Copy over pretrained NADS-Net weights to produces losses that help train the generator
if not os.path.exists('./checkpoints/traditionally_trained_NADS_Net_with_PVT.pth'):
    if os.path.exists('../Data_and_Pretrained_Weights/Pretrained_Weights/latest_net_G.pth'):
        copyfile('../Data_and_Pretrained_Weights/Pretrained_Weights/traditionally_trained_NADS_Net_with_PVT.pth', './checkpoints/traditionally_trained_NADS_Net_with_PVT.pth')
    else:
        sys.exit('Please download the folder Data_and_Pretrained_Weights from OneDrive first and place it at the same level as GAN and NADS-Net_with_PVT.')

# Load the dataset
training_JSON_path = '../Data_and_Pretrained_Weights/NADS_Net_Sample_Data/Train_keypoint_Annotation_sample.json'
raw_images_path = '../Data_and_Pretrained_Weights/NADS_Net_Sample_Data/Raw_Images/'
seatbelt_masks_path = '../Data_and_Pretrained_Weights/NADS_Net_Sample_Data/Seatbelt_Mask/'

num_dataloader_threads = 10
train_dataset = Dataset_Generator_Aisin(training_JSON_path, raw_images_path, seatbelt_masks_path, False, augment=False)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, num_workers=num_dataloader_threads, pin_memory=True, shuffle=True, drop_last=False)
dataset_size = len(train_dataset)
print('Number of training samples = %i' % dataset_size)

# Initialize NADS-Net (used for producing losses to help train Generator)
NADS_Net = NADS_Net(True, True, False).to(torch.device("cuda:1"))
NADS_Net.load_state_dict(torch.load('./checkpoints/traditionally_trained_NADS_Net_with_PVT.pth',  map_location='cuda:1'))

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

        # Create an imput dictionary for the GAN
        data = {}
        data['input_semantics'] = torch.cat((keypoint_heatmaps, PAFs, seatbelt_segmentations), dim=1).type(torch.FloatTensor)
        data['image'] = real_images

        iter_counter.record_one_iteration()

        # Train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data)

        # Train discriminator
        trainer.run_discriminator_one_step(data)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            maxed_input = torch.max(torch.abs(data['input_semantics']), dim=1).values
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

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
