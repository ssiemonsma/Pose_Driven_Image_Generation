class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_gan_options():
    # GAN Options
    opt = Namespace()
    opt.lambda_keypoint = 100
    opt.lambda_paf = 100
    opt.lambda_seatbelt = 1
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
    opt.save_epoch_freq = 10
    opt.continue_train = True
    opt.lr = 0.00002
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

    return opt