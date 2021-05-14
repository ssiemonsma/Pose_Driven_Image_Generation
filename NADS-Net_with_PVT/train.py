import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataset_generator import Dataset_Generator_Aisin
from NADS_Net_model import NADS_Net
from tqdm import tqdm
import numpy as np
import random
import cv2
from trainers.gan_trainer import GAN_Trainer
import os
from util.get_gan_options import get_gan_options
from shutil import copyfile
import sys

# We need to use the first GPU for the generator and the second for NADS-Net
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# Instantiate the Generator
# We're not actually training, but running the generator can be facilitated by its trainer
generator = GAN_Trainer(get_gan_options(), None)

# Set a filename for saving NADS-Net weights
filename = './checkpoints/training_NADS_Net_weights.pth'

# Training Parameters
arm_augment_type = 'slightly'   # Choose from 'slightly', 'highly', or 'none'
start_from_pretrained = True
pretrained_filename = './checkpoints/traditionally_trained_NADS_Net_with_PVT.pth'
num_training_epochs = 100
batch_size = 4
starting_lr = 1e-5
lr_schedule_type = 'metric-based'
lr_gamma = 0.8  # (for both fixed and metric-based scheduler) This is the factor the learning rate decreases by after the metric doesn't improve for some time
patience = 4   # (for metric-based scheduler only) The number of epochs that must pass without metric improvement for the learning rate to decrease
step_size = 20   # (for fixed scheduler only) After this many epochs without improvement, the learning rate is decreased
weight_decay = 5e-7
num_dataloader_threads = 10
include_background_output = False

if start_from_pretrained and pretrained_filename == './checkpoints/traditionally_trained_NADS_Net_with_PVT.pth':
    # Copy over pretrained NADS-Net weights (trained using traditional means)
    if not os.path.exists('./checkpoints/traditionally_trained_NADS_Net_with_PVT.pth'):
        if os.path.exists('../Data_and_Pretrained_Weights/Pretrained_Weights/latest_net_G.pth'):
            copyfile('../Data_and_Pretrained_Weights/Pretrained_Weights/traditionally_trained_NADS_Net_with_PVT.pth', './checkpoints/traditionally_trained_NADS_Net_with_PVT.pth')
        else:
            sys.exit('Please download the folder Data_and_Pretrained_Weights from OneDrive first and place it at the same level as GAN and NADS-Net_with_PVT.')

# Instantiate the Network
device = torch.device("cuda:1")
net = NADS_Net(True, True, include_background_output).to(device)
print('Network contains', sum([p.numel() for p in net.parameters()]), 'parameters.')

# Create the Data Loaders
training_JSON_path = '../Data_and_Pretrained_Weights/NADS_Net_Sample_Data/Train_keypoint_Annotation_sample.json'
validation_JSON_path = '../Data_and_Pretrained_Weights/NADS_Net_Sample_Data/Train_keypoint_Annotation_sample.json'
raw_images_path = '../Data_and_Pretrained_Weights/NADS_Net_Sample_Data/Raw_Images/'
seatbelt_masks_path = '../Data_and_Pretrained_Weights/NADS_Net_Sample_Data/Seatbelt_Mask/'

train_dataset = Dataset_Generator_Aisin(training_JSON_path, raw_images_path, seatbelt_masks_path, include_background_output, augment=True, arm_augment_type=arm_augment_type)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_dataloader_threads, pin_memory=True, shuffle=True, drop_last=True)
num_training_samples = len(train_dataset)
print('Number of training samples = %i' % num_training_samples)

valid_dataset = Dataset_Generator_Aisin(validation_JSON_path, raw_images_path, seatbelt_masks_path, include_background_output, augment=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_dataloader_threads, pin_memory=True, shuffle=False, drop_last=False)
num_validation_samples = len(valid_dataset)
print('Number of validation samples = %i' % num_validation_samples)

# Set up the Optimizer, Loss Function, Learning Rate Scheduler, and Logger
optimizer = optim.SGD(net.parameters(), lr=starting_lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)

if lr_schedule_type == 'fixed':
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_gamma)
elif lr_schedule_type == 'metric-based':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_gamma, patience=patience, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10)

# Note that this "MSE" value is calculated as a sum and is normalized by the batch size
# due to how this loss was calculated in the original NADS-Net code
def MSE_criterion(input, target, batch_size):
    return nn.MSELoss(reduction='sum')(input, target)/batch_size

# This a linear combination of Dice loss and Binary Cross Entropy loss
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE, dice_loss

DiceBCE_criterion = DiceBCELoss()

if start_from_pretrained:
    net.load_state_dict(torch.load(pretrained_filename,  map_location='cuda:1'), strict=False)

# We can't augment input maps inside of the generator since this has to be done after
# generating fake images with the GPU-accelerated Generator.  GPU-accelerated operations
# unfortunately can't be placed in a multi-threaded dataloader.
# The augmentations simply random scaling and translation augmentations.
def augment_maps(img, heat, paf, seat):
    img, heat, paf, seat = list(
        map(lambda x: np.moveaxis(x, 0, 2),
            [img, heat, paf, seat]))

    h, w = 384, 384
    h_mask, w_mask = 96, 96

    npblank = np.ones((384, 384, 3), dtype=np.float64) * 128
    npheat = np.zeros((96, 96, 9), dtype=np.float64)
    nppaf = np.zeros((96, 96, 16), dtype=np.float64)
    npseat = np.zeros((384, 384), dtype=np.uint8)

    scale_h = random.randrange(3, 11) * 0.1
    scale_w = random.randrange(3, 11) * 0.1

    new_h, new_w = int(scale_h * h + 0.5), int(scale_w * w + 0.5)
    heat_h, heat_w = int(scale_h * h_mask + 0.5), int(scale_w * w_mask + 0.5)

    img = cv2.resize(img, (0, 0), fx=(new_w / w), fy=(new_h / h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.resize(heat, (0, 0), fx=(heat_w / w_mask), fy=(heat_h / h_mask), interpolation=cv2.INTER_CUBIC)
    paf = cv2.resize(paf, (0, 0), fx=(heat_w / w_mask), fy=(heat_h / h_mask), interpolation=cv2.INTER_CUBIC)
    seat = cv2.resize(seat, (0, 0), fx=(new_w / w), fy=(new_h / h), interpolation=cv2.INTER_CUBIC)

    dy = 384 - new_h
    dy = random.randrange(0, dy + 1)

    dx = 384 - new_w
    dx = random.randrange(0, dx + 1)

    hdy = round(dy / 4)
    hdx = round(dx / 4)

    npblank[dy:dy + new_h, dx:dx + new_w, :] = img

    npheat[hdy:hdy + heat_h, hdx:hdx + heat_w, :] = heat
    nppaf[hdy:hdy + heat_h, hdx:hdx + heat_w, :] = paf
    npseat[dy:dy + new_h, dx:dx + new_w] = seat
    npseat = npseat[...,np.newaxis]

    npblank, npheat, nppaf, npseat = list(
        map(lambda x: torch.Tensor(np.moveaxis(x, 2, 0)),
            [npblank, npheat, nppaf, npseat]))

    return npblank, npheat, nppaf, npseat

best_loss = np.float('inf')
writer = SummaryWriter()  # Writes training statistics to logs

# Training Loop
for epoch in range(num_training_epochs):
    running_total_loss = 0
    running_keypoint_heatmap_MSE_loss = 0
    running_PAF_MSE_loss = 0
    running_seatbelt_dice_BCE_loss = 0
    running_seatbelt_dice_loss = 0

    progress_bar = tqdm(train_loader)
    for i, data in enumerate(progress_bar):
        input_images, keypoint_heatmap_labels, PAF_labels, seatbelt_labels, keypoint_heatmap_labels_small, PAF_labels_small, keypoint_heatmap_labels_small_noaug, PAF_labels_small_noaug = data[0].to(device), data[1].to(device), data[2].to(device), data[3].float().to(device), data[4].float().to(device), data[5].float().to(device), data[6].float().to(device), data[7].float().to(device)
        seatbelt_labels = torch.unsqueeze(torch.squeeze(seatbelt_labels), 1)

        # Setting half of the small labels to not have limb augmentation
        keypoint_heatmap_labels_small[batch_size//2:-1,:,:,:] = keypoint_heatmap_labels_small_noaug[batch_size//2:-1,:,:,:]
        PAF_labels_small[batch_size//2:-1,:,:,:] = PAF_labels_small_noaug[batch_size//2:-1,:,:,:]

        # Create an imput dictionary for the Generator with half of the training samples
        data = {}
        data['input_semantics'] = torch.cat((keypoint_heatmap_labels[0:batch_size//2,:,:,:], PAF_labels[0:batch_size//2,:,:,:], seatbelt_labels[0:batch_size//2,:,:,:]), dim=1).type(torch.FloatTensor)
        data['image'] = input_images[0:2,:,:,:].type(torch.FloatTensor)  # Not actually used since we're not calculating losses for the generator

        # Replace half of the input images with generated, pose-modified images
        input_images[0:batch_size//2,:,:,:] = generator.run_generator(data)

        # Standard image augmentation (random resizes/translations) occur here
        # since we cannot not place the GPU-accelerated generator operations inside of the multithreaded dataloader
        for j in range(batch_size):
            input_images[j,:,:,:], keypoint_heatmap_labels_small[j,:,:,:], PAF_labels_small[j,:,:,:], seatbelt_labels[j,:,:,:] = augment_maps(input_images[j,:,:,:].detach().cpu().numpy(), keypoint_heatmap_labels_small[j,:,:,:].detach().cpu().numpy(), PAF_labels_small[j,:,:,:].detach().cpu().numpy(), seatbelt_labels[j,:,:,:].detach().cpu().numpy())

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        keypoint_heatmap, PAFs, seatbelt_segmentations = net(input_images)

        # Calculate losses
        keypoint_heatmap_MSE_loss = MSE_criterion(keypoint_heatmap, keypoint_heatmap_labels_small, len(input_images))
        PAF_MSE_loss = MSE_criterion(PAFs, PAF_labels_small, len(input_images))
        seatbelt_MSE_loss = MSE_criterion(seatbelt_segmentations.squeeze(), seatbelt_labels.squeeze(), len(input_images))
        seatbelt_dice_BCE_loss, seatbelt_dice_loss = DiceBCE_criterion(seatbelt_segmentations.squeeze(), seatbelt_labels.squeeze())

        # Note that the seatbelt Dice-BCE loss is scaled to be in the same magnitude range as the MSE values
        total_loss = keypoint_heatmap_MSE_loss + PAF_MSE_loss + seatbelt_dice_BCE_loss*500

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        running_total_loss += total_loss.item()*len(input_images)
        running_keypoint_heatmap_MSE_loss += keypoint_heatmap_MSE_loss.item()*len(input_images)
        running_PAF_MSE_loss += PAF_MSE_loss.item()*len(input_images)
        running_seatbelt_dice_BCE_loss += seatbelt_dice_BCE_loss.item()*len(input_images)
        running_seatbelt_dice_loss += seatbelt_dice_loss.item()*len(input_images)

        progress_bar.set_description('Epoch %i Training' % epoch)
        progress_bar.set_postfix_str('Train Batch MSE: %.3f, Keypoint heatmap MSE: %.3f, PAF MSE: %.3f, Seatbelt Dice-BCE: %.3f, Seatbelt Dice: %.3f' % (total_loss, keypoint_heatmap_MSE_loss, PAF_MSE_loss, seatbelt_dice_BCE_loss, seatbelt_dice_loss))

    epoch_total_loss = running_total_loss/num_training_samples
    epoch_keypoint_heatmap_MSE_loss = running_keypoint_heatmap_MSE_loss/num_training_samples
    epoch_PAF_MSE_loss = running_PAF_MSE_loss/num_training_samples
    epoch_seatbelt_dice_BCE_loss = running_seatbelt_dice_BCE_loss/num_training_samples
    epoch_seatbelt_dice_loss = running_seatbelt_dice_loss/num_training_samples

    # Write metrics to logs
    writer.add_scalar("Loss/training/epochs/total_loss", epoch_total_loss, epoch)
    writer.add_scalar("Loss/training/epochs/keypoint_heatmap_MSE", epoch_keypoint_heatmap_MSE_loss, epoch)
    writer.add_scalar("Loss/training/epochs/PAF_MSE", epoch_PAF_MSE_loss, epoch)
    writer.add_scalar("Loss/training/epochs/seatbelt_dice_BCE_loss", epoch_seatbelt_dice_BCE_loss, epoch)
    writer.add_scalar("Loss/training/epochs/seatbelt_dice_loss", epoch_seatbelt_dice_loss, epoch)

    print('Epoch %i: Training Loss: %.3f, Keypoint heatmap MSE: %.3f, PAF MSE: %.3f, Seatbelt Dice-BCE: %.3f, Seatbelt Dice: %.3f' % (epoch, epoch_total_loss, epoch_keypoint_heatmap_MSE_loss, epoch_PAF_MSE_loss, epoch_seatbelt_dice_BCE_loss, epoch_seatbelt_dice_loss))

    # Validation
    with torch.no_grad():
        running_total_loss = 0
        running_keypoint_heatmap_MSE_loss = 0
        running_PAF_MSE_loss = 0
        running_seatbelt_dice_BCE_loss = 0
        running_seatbelt_dice_loss = 0

        progress_bar = tqdm(valid_loader)
        for i, data in enumerate(progress_bar):
            input_images, keypoint_heatmap_masks, PAF_masks, keypoint_heatmap_labels, PAF_labels, seatbelt_labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].float().to(device), data[4].float().to(device), data[5].float().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            keypoint_heatmap, PAFs, seatbelt_segmentations = net(input_images)

            # Calculate losses
            keypoint_heatmap_MSE_loss = MSE_criterion(keypoint_heatmap, keypoint_heatmap_labels, len(input_images))
            PAF_MSE_loss = MSE_criterion(PAFs, PAF_labels, len(input_images))
            seatbelt_MSE_loss = MSE_criterion(seatbelt_segmentations.squeeze(), seatbelt_labels.squeeze(), len(input_images))
            seatbelt_dice_BCE_loss, seatbelt_dice_loss = DiceBCE_criterion(seatbelt_segmentations.squeeze(), seatbelt_labels.squeeze())

            # Note that the seatbelt Dice-BCE loss is scaled to be in the same magnitude range as the MSE values
            total_loss = keypoint_heatmap_MSE_loss + PAF_MSE_loss + seatbelt_dice_BCE_loss*500

            running_total_loss += total_loss.item()*len(input_images)
            running_keypoint_heatmap_MSE_loss += keypoint_heatmap_MSE_loss.item()*len(input_images)
            running_PAF_MSE_loss += PAF_MSE_loss.item()*len(input_images)
            running_seatbelt_dice_BCE_loss += seatbelt_dice_BCE_loss.item() * len(input_images)
            running_seatbelt_dice_loss += seatbelt_dice_loss.item() * len(input_images)

            progress_bar.set_description('Epoch %i Validation' % epoch)
            progress_bar.set_postfix_str('Valid Batch MSE: %.3f, Keypoint heatmap MSE: %.3f, PAF MSE: %.3f, Seatbelt Dice-BCE: %.3f, Seatbelt Dice: %.3f' % (total_loss, keypoint_heatmap_MSE_loss, PAF_MSE_loss, seatbelt_dice_BCE_loss, seatbelt_dice_loss))

        epoch_total_loss = running_total_loss/num_validation_samples
        epoch_keypoint_heatmap_MSE_loss = running_keypoint_heatmap_MSE_loss/num_validation_samples
        epoch_PAF_MSE_loss = running_PAF_MSE_loss/num_validation_samples
        epoch_seatbelt_dice_BCE_loss = running_seatbelt_dice_BCE_loss/num_validation_samples
        epoch_seatbelt_dice_loss = running_seatbelt_dice_loss/num_validation_samples

        # Write metrics to logs
        writer.add_scalar("Loss/validation/epochs/total_loss", epoch_total_loss, epoch)
        writer.add_scalar("Loss/validation/epochs/keypoint_heatmap_MSE", epoch_keypoint_heatmap_MSE_loss, epoch)
        writer.add_scalar("Loss/validation/epochs/PAF_MSE", epoch_PAF_MSE_loss, epoch)
        writer.add_scalar("Loss/validation/epochs/seatbelt_dice_BCE_loss", epoch_seatbelt_dice_BCE_loss, epoch)
        writer.add_scalar("Loss/validation/epochs/seatbelt_dice_loss", epoch_seatbelt_dice_loss, epoch)

        print('Epoch %i: Validation Loss: %.3f, Keypoint heatmap MSE: %.3f, PAF MSE: %.3f, Seatbelt Dice-BCE: %.3f, Seatbelt Dice: %.3f' % (epoch, epoch_total_loss, epoch_keypoint_heatmap_MSE_loss, epoch_PAF_MSE_loss, epoch_seatbelt_dice_BCE_loss, epoch_seatbelt_dice_loss))

    # Save the network if the validation loss has improved
    if epoch_total_loss < best_loss:
        print('Best validation loss achieved:', epoch_total_loss)
        torch.save(net.state_dict(), filename)
        best_loss = epoch_total_loss

    if lr_schedule_type == 'fixed':
        scheduler.step()
    elif lr_schedule_type == 'metric-based':
        scheduler.step(epoch_total_loss)
