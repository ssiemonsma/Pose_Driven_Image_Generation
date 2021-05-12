"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import os
import torch
from dataset_generator import Dataset_Generator_Aisin
from model import NADS_Net

torch.set_default_tensor_type(torch.FloatTensor)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4,5"

NADS_Net = NADS_Net(True, True, False).to(torch.device("cuda:1"))
NADS_Net.load_state_dict(torch.load('weights_training_with_tiny_PVT.pth',  map_location='cuda:1'))



sys.argv = ["train.py", "--name", "label2city_512p", "--label_nc", "0", "--no_instance", "--netG", "SPADE"]
# sys.argv = ["train.py", "--name", "label2city_512p", "--label_nc", "0", "--no_instance", "--netG", "Pix2PixHD"]
# sys.argv = ["train.py", "--name", "label2city_512p", "--label_nc", "0", "--input_nc", "26", "load_size", "384", "crop_size", "384", "display_winsize", "384", "--no_instance", "--netG", "Pix2PixHD"]


# parse options
opt = TrainOptions().parse()
# parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
# parser.set_defaults(preprocess_mode='resize_and_crop')
# load_size = 286 if is_train else 256
# parser.set_defaults(load_size=load_size)
# parser.set_defaults(crop_size=256)
# parser.set_defaults(display_winsize=256)
# parser.set_defaults(label_nc=13)
# parser.set_defaults(contain_dontcare_label=False)
opt.gpu_ids = [0]
opt.ngf = 64
opt.crop_size = 256
opt.display_winsize = 256
opt.semantic_nc = 26
opt.num_D = 2
opt.output_nc = 3
opt.contain_dontcare_label = False
# opt.dataset_mode = ''
opt.save_latest_freq = 500
opt.print_freq = 100
opt.display_freq = 100
opt.continue_train = True



# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
training_JSON_path = '/localscratch/Users/ssiemons/NADS-Net/Aisin_Dataset/Train_keypoint_Annotations.json'
validation_JSON_path = '/localscratch/Users/ssiemons/NADS-Net/Aisin_Dataset/Test_keypoint_Annotations.json'
raw_images_path = '/localscratch/Users/ssiemons/NADS-Net/Aisin_Dataset/Raw_Images/'
seatbelt_masks_path = '/localscratch/Users/ssiemons/NADS-Net/Aisin_Dataset/Seatbelt_Mask/'
batch_size = 1
num_dataloader_threads = 10
train_dataset = Dataset_Generator_Aisin(training_JSON_path, raw_images_path, seatbelt_masks_path, False, augment=False)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_dataloader_threads, pin_memory=True, shuffle=True, drop_last=False)
dataset_size = len(train_dataset)
print('Number of training samples = %i' % dataset_size)

# create trainer for our model
trainer = Pix2PixTrainer(opt, NADS_Net)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))


# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data in enumerate(dataloader, start=iter_counter.epoch_iter):
        input_images, keypoint_heatmap_labels, PAF_labels, seatbelt_labels, keypoint_heatmap_labels_small, PAF_labels_small = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].float().cuda(), data[4].float().cuda(), data[5].float().cuda()
        # input_images, keypoint_heatmap_masks, PAF_masks, keypoint_heatmap_labels, PAF_labels, seatbelt_labels = data[0].cpu(), data[1].cpu(), data[2].cpu(), data[3].float().cpu(), data[4].float().cpu(), data[5].float().cpu()
        seatbelt_labels = seatbelt_labels[:,:,:,:,0]

        # plt.imshow(np.moveaxis(input_images.cpu().numpy()[0, :, :, :], 0, 2))
        # plt.show()
        #
        # plt.imshow(keypoint_heatmap_labels[0, 0, :, :].cpu().numpy())
        # plt.show()
        #
        # plt.imshow(PAF_labels[0, 0, :, :].cpu().numpy())
        # plt.show()

        data = {}

        data['label'] = torch.cat((keypoint_heatmap_labels, PAF_labels, seatbelt_labels), dim=1).type(torch.FloatTensor)
        # data['label'] = keypoint_heatmap_labels[:,0:1,:,:]
        data['image'] = input_images
        data['instance'] = torch.Tensor([0])
        data['path'] = 'fake_path'

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
            maxed_input = torch.max(data['label'], dim=1).values
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
