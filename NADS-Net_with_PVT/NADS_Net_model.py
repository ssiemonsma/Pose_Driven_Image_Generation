import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from PVT.pvt import pvt_tiny

class FPN(nn.Module):
    def __init__(self, num_channels=256):
        super(FPN, self).__init__()

        # self.lateral_layer_5 = nn.Conv2d(2048, num_channels, kernel_size=1, stride=1, padding=0)
        # self.lateral_layer_4 = nn.Conv2d(1024, num_channels, kernel_size=1, stride=1, padding=0)
        # self.lateral_layer_3 = nn.Conv2d(512, num_channels, kernel_size=1, stride=1, padding=0)
        # self.lateral_layer_2 = nn.Conv2d(256, num_channels, kernel_size=1, stride=1, padding=0)
        # embed_dims = [64, 128, 256, 512]

        self.lateral_layer_5 = nn.Conv2d(512, num_channels, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_4 = nn.Conv2d(320, num_channels, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_3 = nn.Conv2d(128, num_channels, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_2 = nn.Conv2d(64, num_channels, kernel_size=1, stride=1, padding=0)

        self.output_layer_5 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.output_layer_4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.output_layer_3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.output_layer_2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def _upsample_like(self, source, target):
        _, _, height, width = target.shape
        return F.interpolate(source, size=(height, width), mode='nearest')

    def forward(self, C2, C3, C4, C5):
        # No ReLu???

        P5 = nn.ReLU(inplace=True)(self.lateral_layer_5(C5))
        P5_upsampled = self._upsample_like(P5, C4)
        P5 = self.output_layer_5(P5)

        P4 = nn.ReLU(inplace=True)(self.lateral_layer_4(C4))
        P4_upsampled = self._upsample_like(P4, C3)
        P3 = self.output_layer_3(P4 + P5_upsampled)

        P3= nn.ReLU(inplace=True)(self.lateral_layer_3(C3))
        P3_upsampled = self._upsample_like(P3, C2)
        P3 = self.output_layer_3(P3 + P4_upsampled)

        P2 = nn.ReLU(inplace=True)(self.lateral_layer_2(C2))
        P2 = self.output_layer_2(P2 + P3_upsampled)

        return P2, P3, P4, P5

class Map_Branch(nn.Module):
    def __init__(self, num_channels_output):
        super(Map_Branch, self).__init__()

        self.convs_on_P5 = self._two_conv()
        self.convs_on_P4 = self._two_conv()
        self.convs_on_P3 = self._two_conv()
        self.convs_on_P2 = self._two_conv()

        self.convs_on_feature_maps = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_channels_output, kernel_size=1, stride=1, padding=0)
        )

    def _two_conv(self):
        return nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, P2, P3, P4, P5):
        D5 = F.interpolate(self.convs_on_P5(P5), scale_factor=8, mode='bilinear')
        D4 = F.interpolate(self.convs_on_P4(P4), scale_factor=4, mode='bilinear')
        D3 = F.interpolate(self.convs_on_P3(P3), scale_factor=2, mode='bilinear')
        D2 = self.convs_on_P2(P2)

        feature_maps = torch.cat((D2, D3, D4, D5), dim=1)

        return self.convs_on_feature_maps(feature_maps)

# Using 2 interpolations
class Segmentation_Branch(nn.Module):
    def __init__(self):
        super(Segmentation_Branch, self).__init__()

        self.convs_on_P5 = self._two_conv()
        self.convs_on_P4 = self._two_conv()
        self.convs_on_P3 = self._two_conv()
        self.convs_on_P2 = self._two_conv()

        self.conv_on_feature_maps = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final_conv1 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.final_conv2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def _two_conv(self):
        return nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, P2, P3, P4, P5):
        D5 = F.interpolate(self.convs_on_P5(P5), scale_factor=8, mode='bilinear')
        D4 = F.interpolate(self.convs_on_P4(P4), scale_factor=4, mode='bilinear')
        D3 = F.interpolate(self.convs_on_P3(P3), scale_factor=2, mode='bilinear')
        D2 = self.convs_on_P2(P2)

        feature_maps = torch.cat((D2, D3, D4, D5), dim=1)

        segmentation = F.interpolate(self.conv_on_feature_maps(feature_maps), scale_factor=2, mode='bilinear')
        segmentation = self.final_conv1(segmentation)
        segmentation = F.interpolate(segmentation, scale_factor=2, mode='bilinear')
        segmentation = self.final_conv2(segmentation)
        segmentation = (F.tanh(segmentation) + 1)/2

        return segmentation

class NADS_Net(torch.nn.Module):
    def __init__(self, include_seatbelt_branch, using_Aisin_output_format, include_background_output):
        super(NADS_Net, self).__init__()

        self.include_seatbelt_branch = include_seatbelt_branch

        if using_Aisin_output_format:
            if include_background_output:
                num_keypoints_output_layers = 10
            else:
                num_keypoints_output_layers = 9

            num_PAF_output_layers = 16
        else:
            num_keypoints_output_layers = 19
            num_PAF_output_layers = 38

        # self.resnet50_modules = nn.ModuleList(list(resnet50(pretrained=True).children())[:-2]).eval()
        self.FPN = FPN()
        self.PVT = pvt_tiny()
        self.keypoint_heatmap_branch = Map_Branch(num_keypoints_output_layers)
        self.PAF_branch = Map_Branch(num_PAF_output_layers)

        if include_seatbelt_branch:
            self.seatbelt_segmentation_branch = Segmentation_Branch()

        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                m.bias.data.fill_(0.0)

        self.FPN.apply(init_weights)
        self.keypoint_heatmap_branch.apply(init_weights)
        self.PAF_branch.apply(init_weights)

        if include_seatbelt_branch:
            self.seatbelt_segmentation_branch.apply(init_weights)

    def forward(self, x):
        # resnet50_outputs = []
        # for i, model in enumerate(self.resnet50_modules):
        #     x = model(x)
        #     if i in [4, 5, 6, 7]:
        #         resnet50_outputs.append(x)
        # C2, C3, C4, C5 = resnet50_outputs
        #
        # P2, P3, P4, P5 = self.FPN(C2, C3, C4, C5)

        C2, C3, C4, C5 = self.PVT(x)
        P2, P3, P4, P5 = self.FPN(C2, C3, C4, C5)


        keypoint_heatmaps = self.keypoint_heatmap_branch(P2, P3, P4, P5)
        PAFs = self.PAF_branch(P2, P3, P4, P5)

        # keypoint_heatmaps *= keypoint_heatmap_masks
        # PAFs *= PAF_masks

        if self.include_seatbelt_branch:
            seatbelt_segmentation = self.seatbelt_segmentation_branch(P2, P3, P4, P5)
            return keypoint_heatmaps, PAFs, seatbelt_segmentation
        else:
            return keypoint_heatmaps, PAFs