import h5py
import json
import numpy as np
import cv2
from torchvision import transforms
import random
from collections import defaultdict
import math
import torch

def distance(a, b):
    if a is not None and b is not None:
        distance = np.linalg.norm(np.asarray(a) - np.asarray(b))
    else:
        distance = 0

    return distance

def calculate_torso_height(joints):
    torso_height = (distance(joints[3], joints[7]) + distance(joints[3], joints[7]))/2
    return torso_height

class Dataset_Generator_Aisin():
    def __init__(self, JSON_path, raw_images_path, seatbelt_masks_path, include_background_output, augment=True, arm_augment_type='none'):
        self.JSON_path = JSON_path
        self.raw_images_path = raw_images_path
        self.seatbelt_masks_path = seatbelt_masks_path
        self.include_background_output = include_background_output
        self.augment = augment
        self.arm_augment_type = arm_augment_type

        dataset = json.load(open(JSON_path, 'r'))

        Ann_list = defaultdict(list)
        for ann in dataset['annotations']:
            if ann['label'] == 'Person':
                Ann_list[ann['image_id']].append(ann['keypoints'])

        self.aisin_data = Ann_list
        self.image_ids = []
        self.all_joints = []

        for ids in self.aisin_data:
            keypoints = self.aisin_data[ids]
            all_keypoints_list = []

            for keypoint in keypoints:
                keypoints_list = []
                kp = np.array(keypoint)
                xs = kp[0::3]*384.0//1920.0
                ys = kp[1::3]*384.0//1080.0
                vs = kp[2::3]

                for idx, (x, y, v) in enumerate(zip(xs, ys, vs)):
                    # only visible and occluded keypoints are used
                    if v >= 1 and x >=0 and y >= 0:# and x < 1080 and y < 1920:
                        keypoints_list.append((x, y))
                    else:
                        keypoints_list.append(None)

                all_keypoints_list.append(keypoints_list)

            self.image_ids.append(ids)
            self.all_joints.append(all_keypoints_list)

        self.PAF_mask = np.ones((16, 96, 96), dtype=np.uint8)

        if include_background_output:
            self.keypoint_heatmap_mask = np.ones((10, 96, 96), dtype=np.uint8)
        else:
            self.keypoint_heatmap_mask = np.ones((9, 96, 96), dtype=np.uint8)

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def augment_maps(self, img, heat, paf, seat):
        h, w = 384, 384
        h_mask, w_mask = 96, 96

        npblank = np.ones((384, 384, 3), dtype=np.float64)*128
        npheat = np.zeros((96, 96, 10), dtype=np.float64)
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

        return npblank, npheat, nppaf, npseat

    def create_keypoint_heatmap(self, num_maps, height, width, all_joints, sigma, stride):

        heatmap = np.zeros((height, width, num_maps), dtype=np.float64)

        for joints in all_joints:
            for plane_idx, joint in enumerate(joints):
                if joint:
                    self._put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride)

        # background
        heatmap[:, :, -1] = np.clip(1.0 - np.amax(heatmap, axis=2), 0.0, 1.0)

        return heatmap

    def create_PAF(self, num_maps, height, width, all_joints, threshold, stride):
        joint_pairs = list(zip(
            [3, 2, 1, 3, 4, 5, 3, 3],
            [2, 1, 0, 4, 5, 6, 7, 8]))

        vectormap = np.zeros((height, width, num_maps * 2), dtype=np.float64)
        countmap = np.zeros((height, width, num_maps), dtype=np.uint8)
        for joints in all_joints:
            for plane_idx, (j_idx1, j_idx2) in enumerate(joint_pairs):
                center_from = joints[j_idx1]
                center_to = joints[j_idx2]

                # skip if no valid pair of keypoints
                if center_from is None or center_to is None:
                    continue

                x1, y1 = (center_from[0] / stride, center_from[1] / stride)
                x2, y2 = (center_to[0] / stride, center_to[1] / stride)

                self._put_paf_on_plane(vectormap, countmap, plane_idx, x1, y1, x2, y2, threshold, height, width)

        return vectormap

    def _put_heatmap_on_plane(self, heatmap, plane_idx, joint, sigma, height, width, stride):
        start = stride / 2.0 - 0.5

        center_x, center_y = joint

        for g_y in range(height):
            for g_x in range(width):
                x = start + g_x * stride
                y = start + g_y * stride
                d2 = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)
                exponent = d2 / 2.0 / sigma / sigma
                if exponent > 4.6052:
                    continue

                heatmap[g_y, g_x, plane_idx] += math.exp(-exponent)
                if heatmap[g_y, g_x, plane_idx] > 1.0:
                    heatmap[g_y, g_x, plane_idx] = 1.0

    def _put_paf_on_plane(self, vectormap, countmap, plane_idx, x1, y1, x2, y2, threshold, height, width):
        min_x = max(0, int(round(min(x1, x2) - threshold)))
        max_x = min(width, int(round(max(x1, x2) + threshold)))

        min_y = max(0, int(round(min(y1, y2) - threshold)))
        max_y = min(height, int(round(max(y1, y2) + threshold)))

        vec_x = x2 - x1
        vec_y = y2 - y1

        norm = math.sqrt(vec_x**2 + vec_y**2)
        if norm < 1e-8:
            return

        vec_x /= norm
        vec_y /= norm

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                bec_x = x - x1
                bec_y = y - y1
                dist = abs(bec_x*vec_y - bec_y*vec_x)

                if dist > threshold:
                    continue

                cnt = countmap[y][x][plane_idx]

                if cnt == 0:
                    vectormap[y][x][plane_idx*2 + 0] = vec_x
                    vectormap[y][x][plane_idx*2 + 1] = vec_y
                else:
                    vectormap[y][x][plane_idx*2 + 0] = (vectormap[y][x][plane_idx*2 + 0]*cnt + vec_x)/(cnt + 1)
                    vectormap[y][x][plane_idx*2 + 1] = (vectormap[y][x][plane_idx*2 + 1]*cnt + vec_y)/(cnt + 1)

                countmap[y][x][plane_idx] += 1

    def __getitem__(self, index):
        img_path = self.image_ids[index].replace('\\', '/')
        joints = self.all_joints[index]

        original_joints = joints
        if self.augment:
            if self.arm_augment_type == 'slightly':
                # Augmenting arm positions
                augment_elbow_max = 0.2
                augment_wrist_max = 0.2
                for i, person in enumerate(joints):
                    torso_height = calculate_torso_height(person)
                    # Check if they are in the front row of the car
                    if torso_height > 70:
                        upper_arm_length = torso_height*.77
                        forearm_length = upper_arm_length*.9

                        # Augment right elbow
                        if person[1] is not None:
                            x_aug = np.round(np.random.uniform(-augment_elbow_max*upper_arm_length/2, augment_elbow_max*upper_arm_length/2))
                            y_aug = np.round(np.random.uniform(-augment_elbow_max*upper_arm_length/2, augment_elbow_max*upper_arm_length/2))
                            joints[i][1] = (person[1][0] + x_aug, person[1][1] + y_aug)
                        # Augment right wrist
                        if person[0] is not None:
                            x_aug = np.round(np.random.uniform(-augment_wrist_max*forearm_length/2, augment_wrist_max*forearm_length/2))
                            y_aug = np.round(np.random.uniform(-augment_wrist_max*forearm_length/2, augment_wrist_max*forearm_length/2))
                            joints[i][0] = (person[0][0] + x_aug, person[0][1] + y_aug)
                        # Augment left elbow
                        if person[5] is not None:
                            x_aug = np.round(np.random.uniform(-augment_elbow_max*upper_arm_length/2, augment_elbow_max*upper_arm_length/2))
                            y_aug = np.round(np.random.uniform(-augment_elbow_max*upper_arm_length/2, augment_elbow_max*upper_arm_length/2))
                            joints[i][5] = (person[5][0] + x_aug, person[5][1] + y_aug)
                        # Augment left wrist
                        if person[6] is not None:
                            x_aug = np.round(np.random.uniform(-augment_wrist_max*forearm_length/2, augment_wrist_max*forearm_length/2))
                            y_aug = np.round(np.random.uniform(-augment_wrist_max*forearm_length/2, augment_wrist_max*forearm_length/2))
                            joints[i][6] = (person[6][0] + x_aug, person[6][1] + y_aug)
            elif self.arm_augment_type == 'highly':
                # Augmenting arm positions
                for i, person in enumerate(joints):
                    torso_height = calculate_torso_height(person)
                    # Check if they are in the front row of the car
                    if torso_height > 70:
                        upper_arm_length = torso_height * .77
                        forearm_length = upper_arm_length * .9

                        if person[2] is not None:
                            # Augment right elbow
                            direction = np.random.normal(size=2)
                            direction = direction / np.linalg.norm(direction)

                            length = np.abs(np.clip(np.random.normal(0.7 * upper_arm_length, 0.2 * upper_arm_length),
                                                    0.1 * upper_arm_length, upper_arm_length))
                            aug = direction * length

                            new_x = np.round(np.clip(person[2][0] + aug[0], 0, 384))
                            new_y = np.round(np.clip(person[2][1] + aug[1], 0, 384))
                            joints[i][1] = (new_x, new_y)

                            # Augment right wrist
                            direction = np.random.normal(size=2)
                            direction = direction / np.linalg.norm(direction)

                            length = np.abs(np.clip(np.random.normal(0.7 * forearm_length, 0.2 * forearm_length),
                                                    0.1 * forearm_length, forearm_length))
                            aug = direction * length

                            new_x = np.round(np.clip(joints[i][1][0] + aug[0], 0, 384))
                            new_y = np.round(np.clip(joints[i][1][1] + aug[1], 0, 384))
                            joints[i][0] = (new_x, new_y)

                        if person[4] is not None:
                            # Augment left elbow
                            direction = np.random.normal(size=2)
                            direction = direction / np.linalg.norm(direction)

                            length = np.abs(np.clip(np.random.normal(0.7 * upper_arm_length, 0.2 * upper_arm_length),
                                                    0.1 * upper_arm_length, upper_arm_length))
                            aug = direction * length

                            new_x = np.round(np.clip(person[4][0] + aug[0], 0, 384))
                            new_y = np.round(np.clip(person[4][1] + aug[1], 0, 384))
                            joints[i][5] = (new_x, new_y)

                            # Augment left wrist
                            direction = np.random.normal(size=2)
                            direction = direction / np.linalg.norm(direction)

                            length = np.abs(np.clip(np.random.normal(0.7 * forearm_length, 0.2 * forearm_length),
                                                    0.1 * forearm_length, forearm_length))
                            aug = direction * length

                            new_x = np.round(np.clip(joints[i][5][0] + aug[0], 0, 384))
                            new_y = np.round(np.clip(joints[i][5][1] + aug[1], 0, 384))
                            joints[i][6] = (new_x, new_y)

        image = open((self.raw_images_path + img_path + '.png'), 'rb').read()
        seatbelt_label = open((self.seatbelt_masks_path + img_path + '.png'), 'rb').read()

        if not image:
            raise Exception('image not read, path=%s' % img_path)

        scale_x = 384/1920
        scale_y = 384/1080

        image = np.fromstring(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)

        image = self.preprocess(np.flip(image, 2)/255)
        image = image.numpy()
        image = np.rollaxis(image, 0, 3)

        seatbelt_label = np.fromstring(seatbelt_label, np.uint8)
        seatbelt_label = cv2.imdecode(seatbelt_label, cv2.IMREAD_COLOR)
        seatbelt_label = cv2.resize(seatbelt_label, (0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
        seatbelt_label = seatbelt_label[:, :, 0]/255.0
        seatbelt_label = seatbelt_label[:, :, np.newaxis]

        if self.augment:
            keypoint_heatmap_labels_small_noaug = self.create_keypoint_heatmap(10, 96, 96, original_joints, 7.0, stride=4)
            PAF_labels_small_noaug = self.create_PAF(8, 96, 96, original_joints, 1, stride=4)
            keypoint_heatmap_labels_small_noaug = keypoint_heatmap_labels_small_noaug[:, :, 0:9]

            keypoint_heatmap_labels = self.create_keypoint_heatmap(10, 384, 384, joints, 7.0, stride=1)
            PAF_labels = self.create_PAF(8, 384, 384, joints, 1, stride=1)
            keypoint_heatmap_labels_small = self.create_keypoint_heatmap(10, 96, 96, joints, 7.0, stride=4)
            PAF_labels_small = self.create_PAF(8, 96, 96, joints, 1, stride=4)
            keypoint_heatmap_labels_small = keypoint_heatmap_labels_small[:,:,0:9]
        else:
            keypoint_heatmap_labels = self.create_keypoint_heatmap(10, 96, 96, joints, 7.0, stride=4)
            PAF_labels = self.create_PAF(8, 96, 96, joints, 1, stride=4)

        # Move the channel dimension to the correct PyTorch position
        if self.augment:
            image, keypoint_heatmap_labels, PAF_labels, keypoint_heatmap_labels_small, PAF_labels_small, keypoint_heatmap_labels_small_noaug, PAF_labels_small_noaug = list(map(lambda x: np.rollaxis(x, 2), [image, keypoint_heatmap_labels, PAF_labels, keypoint_heatmap_labels_small, PAF_labels_small, keypoint_heatmap_labels_small_noaug, PAF_labels_small_noaug]))
        else:
            image, keypoint_heatmap_labels, PAF_labels = list(map(lambda x: np.rollaxis(x, 2), [image, keypoint_heatmap_labels, PAF_labels]))

        image = image.astype(np.float32)

        seatbelt_label = seatbelt_label.astype(np.uint8)
        seatbelt_label = seatbelt_label[np.newaxis, :, :]

        if not self.include_background_output:
            keypoint_heatmap_labels = keypoint_heatmap_labels[:9,:,:]

        if self.augment:
            return image, keypoint_heatmap_labels, PAF_labels, seatbelt_label, keypoint_heatmap_labels_small, PAF_labels_small, keypoint_heatmap_labels_small_noaug, PAF_labels_small_noaug
        else:
            return image, self.keypoint_heatmap_mask, self.PAF_mask, keypoint_heatmap_labels, PAF_labels, seatbelt_label

    def __len__(self):
        return len(self.image_ids)

class CanonicalConfig:
    def __init__(self):
        self.width = 384
        self.height = 384

        self.stride = 4

        self.parts = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank", "Reye", "Leye", "Rear", "Lear"]
        self.num_parts = len(self.parts)
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))
        self.parts += ["background"]
        self.num_parts_with_background = len(self.parts)

        leftParts, rightParts = CanonicalConfig.ltr_parts(self.parts_dict)
        self.leftParts = leftParts
        self.rightParts = rightParts

        # this numbers probably copied from matlab they are 1.. based not 0.. based
        self.limb_from =  ['neck', 'Rhip', 'Rkne', 'neck', 'Lhip', 'Lkne', 'neck', 'Rsho', 'Relb', 'Rsho', 'neck', 'Lsho', 'Lelb', 'Lsho',
         'neck', 'nose', 'nose', 'Reye', 'Leye']
        self.limb_to = ['Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Rsho', 'Relb', 'Rwri', 'Rear', 'Lsho', 'Lelb', 'Lwri', 'Lear',
         'nose', 'Reye', 'Leye', 'Rear', 'Lear']

        self.limb_from = [ self.parts_dict[n] for n in self.limb_from ]
        self.limb_to = [ self.parts_dict[n] for n in self.limb_to ]

        assert self.limb_from == [x-1 for x in [2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16]]
        assert self.limb_to == [x-1 for x in [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]]

        self.limbs_conn = list(zip(self.limb_from, self.limb_to))

        self.paf_layers = 2*len(self.limbs_conn)
        self.heat_layers = self.num_parts
        self.num_layers = self.paf_layers + self.heat_layers + 1

        self.paf_start = 0
        self.heat_start = self.paf_layers
        self.bkg_start = self.paf_layers + self.heat_layers

        #self.data_shape = (self.height, self.width, 3)     # 368, 368, 3
        self.mask_shape = (self.height//self.stride, self.width//self.stride)  # 46, 46
        self.parts_shape = (self.height//self.stride, self.width//self.stride, self.num_layers)  # 46, 46, 57

        class TransformationParams:
            def __init__(self):
                self.target_dist = 0.6;
                self.scale_prob = 1;  # TODO: this is actually scale unprobability, i.e. 1 = off, 0 = always, not sure if it is a bug or not
                self.scale_min = 0.5;
                self.scale_max = 1.1;
                self.max_rotate_degree = 40.
                self.center_perterb_max = 40.
                self.flip_prob = 0.5
                self.sigma = 7.
                self.paf_thre = 8.  # it is original 1.0 * stride in this program

        self.transform_params = TransformationParams()

    @staticmethod
    def ltr_parts(parts_dict):
        # when we flip image left parts became right parts and vice versa. This is the list of parts to exchange each other.
        leftParts  = [ parts_dict[p] for p in ["Lsho", "Lelb", "Lwri", "Lhip", "Lkne", "Lank", "Leye", "Lear"] ]
        rightParts = [ parts_dict[p] for p in ["Rsho", "Relb", "Rwri", "Rhip", "Rkne", "Rank", "Reye", "Rear"] ]
        return leftParts, rightParts


class COCOSourceConfig:
    def __init__(self, hdf5_source):
        self.hdf5_source = hdf5_source
        self.parts = ['nose', 'Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
             'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']

        self.num_parts = len(self.parts)

        # for COCO neck is calculated like mean of 2 shoulders.
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))

    def convert(self, meta, global_config):
        joints = np.array(meta['joints'])

        assert joints.shape[1] == len(self.parts)

        result = np.zeros((joints.shape[0], global_config.num_parts, 3), dtype=np.float)
        result[:,:,2]=3.  # OURS - # 3 never marked up in this dataset, 2 - not marked up in this person, 1 - marked and visible, 0 - marked but invisible

        for p in self.parts:
            coco_id = self.parts_dict[p]

            if p in global_config.parts_dict:
                global_id = global_config.parts_dict[p]
                assert global_id!=1, "neck shouldn't be known yet"
                result[:,global_id,:]=joints[:,coco_id,:]

        if 'neck' in global_config.parts_dict:
            neckG = global_config.parts_dict['neck']
            RshoC = self.parts_dict['Rsho']
            LshoC = self.parts_dict['Lsho']

            # no neck in coco database, we calculate it as average of shoulders
            # TODO: we use 0 - hidden, 1 visible, 2 absent - it is not coco values they processed by generate_hdf5
            both_shoulders_known = (joints[:, LshoC, 2]<2)  &  (joints[:, RshoC, 2] < 2)

            result[~both_shoulders_known, neckG, 2] = 2. # otherwise they will be 3. aka 'never marked in this dataset'

            result[both_shoulders_known, neckG, 0:2] = (joints[both_shoulders_known, RshoC, 0:2] +
                                                        joints[both_shoulders_known, LshoC, 0:2]) / 2
            result[both_shoulders_known, neckG, 2] = np.minimum(joints[both_shoulders_known, RshoC, 2],
                                                                     joints[both_shoulders_known, LshoC, 2])

        meta['joints'] = result

        return meta

    def convert_mask(self, mask, global_config, joints = None):
        mask = np.repeat(mask[:,:,np.newaxis], global_config.num_layers, axis=2)
        return mask

    def source(self):
        return self.hdf5_source