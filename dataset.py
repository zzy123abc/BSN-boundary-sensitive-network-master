# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pandas
import numpy
import json
import torch.utils.data as data
import os
import torch
import h5py
import pdb
from sklearn.preprocessing import normalize

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

class VideoDataSet(data.Dataset):
    def __init__(self, opt, subset="train"):
        self.temporal_scale = opt["temporal_scale"]
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = opt["mode"]
        self.feature_path = opt["feature_path"]
        self.boundary_ratio = opt["boundary_ratio"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self.training_anno = load_json('/gruntdata/disk1/data/captions/train.json')
        self.validation_anno = load_json('/gruntdata/disk1/data/captions/val_1.json')
        self._getDatasetDict()

    def _getDatasetDict(self):
        json_data = load_json('Evaluation/data/activity_net.v1-3.min.json')
        database = json_data['database']
        self.video_dict = {}
        for i in range(len(database.keys())):
            video_name = database.keys()[i]
            video_info = database[video_name]
            video_subset = database[video_name]['subset']
            if self.subset == "full":
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                if "v_" + video_name in self.training_anno.keys():
                    self.video_dict[video_name] = video_info
                if "v_" + video_name in self.validation_anno.keys():
                    self.video_dict[video_name] = video_info
        self.video_list = self.video_dict.keys()
        print "%s subset video numbers: %d" % (self.subset, len(self.video_list))

    def __getitem__(self, index):
        video_data, anchor_xmin, anchor_xmax = self._get_base_data(index)
        if self.mode == "train":
            # video_data,anchor_xmin,anchor_xmax = self._get_base_data(index)
            match_score_action, match_score_start, match_score_end = self._get_train_label(index, anchor_xmin,
                                                                                           anchor_xmax)
            return video_data, match_score_action, match_score_start, match_score_end
        else:
            # video_name = self.video_list[index]
            # video_info = self.video_dict[video_name]
            # video_second = video_info['duration']
            # if video_second > 25:
            #     video_data, anchor_xmin, anchor_xmax = self._get_base_data(index)
            # else:
            #     video_data, anchor_xmin, anchor_xmax = self._get_original_data(index)
            # video_data, anchor_xmin, anchor_xmax = self._get_original_data(index)
            return index, video_data, anchor_xmin, anchor_xmax

    def l2_normalize(self, video_df):
        video_data = video_df.values[:, :]
        video_data = np.array(video_data)
        video_data = normalize(video_data, norm='l2')
        video_data = pd.DataFrame(video_data)
        return video_data

    def _get_base_data(self, index):
        video_name = self.video_list[index]
        anchor_xmin = [self.temporal_gap * i for i in range(self.temporal_scale)]
        anchor_xmax = [self.temporal_gap * i for i in range(1, self.temporal_scale + 1)]
        video_df2 = pd.read_csv(
            self.feature_path + "resnet_csv_mean_" + str(self.temporal_scale) + "/" + "v_" + video_name + ".csv")
        video_df3 = pd.read_csv(
            self.feature_path + "audio_csv_mean_" + str(self.temporal_scale) + "/" + "v_" + video_name + ".csv")
        video_df3 = self.l2_normalize(video_df3)
        video_df = [video_df2, video_df3]
        video_df = pd.concat(video_df, axis=1)
        video_data = video_df.values[:, :]
        video_data = torch.Tensor(video_data)
        video_data = torch.transpose(video_data, 0, 1)
        video_data.float()
        return video_data, anchor_xmin, anchor_xmax

    def readData(self, video_name):
        # dir = "/gruntdata/disk1/data/c3d/sub_activitynet_v1-3.c3d.hdf5"
        dir = "/gruntdata/disk1/data/resnet152_features_activitynet_5fps_320x240.hdf5"
        f = h5py.File(dir, 'r')
        video_name = "v_" + video_name
        # new_data = f[video_name]['c3d_features']
        new_data = f[video_name]
        return new_data[:]

    def _get_original_data(self, index):
        video_name = self.video_list[index]
        video_data = self.readData(video_name)
        anchor_xmin = [1. / len(video_data) * i for i in range(len(video_data))]
        anchor_xmax = [1. / len(video_data) * i for i in range(1, len(video_data) + 1)]
        video_data = torch.Tensor(video_data)
        video_data = torch.transpose(video_data, 0, 1)
        video_data.float()
        return video_data, anchor_xmin, anchor_xmax

    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        video_name = self.video_list[index]
        video_info = self.video_dict[video_name]
        video_second = video_info['duration']
        corrected_second = video_second
        video_name = "v_" + video_name
        if self.subset == 'training':
            video_labels = self.training_anno[video_name]['timestamps']
        if self.subset == 'validation':
            video_labels = self.validation_anno[video_name]['timestamps']

        gt_bbox = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info[0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info[1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])

        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]

        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        match_score_action = []
        for jdx in range(len(anchor_xmin)):
            match_score_action.append(
                np.max(self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_xmins, gt_xmaxs)))
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_action = torch.Tensor(match_score_action)
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        return match_score_action, match_score_start, match_score_end

    def _ioa_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)
        return scores

    def __len__(self):
        return len(self.video_list)


class ProposalDataSet(data.Dataset):
    def __init__(self, opt, subset="train"):

        self.subset = subset
        self.mode = opt["mode"]
        if self.mode == "train":
            self.top_K = opt["pem_top_K"]
        else:
            self.top_K = opt["pem_top_K_inference"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self.training_anno = load_json('/gruntdata/disk1/data/captions/train.json')
        self.validation_anno = load_json('/gruntdata/disk1/data/captions/val_1.json')
        self._getDatasetDict()

    def _getDatasetDict(self):
        json_data = load_json('Evaluation/data/activity_net.v1-3.min.json')
        database = json_data['database']
        self.video_dict = {}
        for i in range(len(database.keys())):
            video_name = database.keys()[i]
            video_info = database[video_name]
            video_subset = database[video_name]['subset']
            if self.subset == "full":
                self.video_dict[video_name] = video_info
            if self.subset == "testing" and video_subset == "testing":
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                if "v_" + video_name in self.training_anno.keys():
                    self.video_dict[video_name] = video_info
                if "v_" + video_name in self.validation_anno.keys():
                    self.video_dict[video_name] = video_info
        self.video_list = self.video_dict.keys()
        print "%s subset video numbers: %d" % (self.subset, len(self.video_list))

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_name = self.video_list[index]
        pdf = pandas.read_csv("./output/PGM_proposals/" + video_name + ".csv")
        pdf = pdf[:self.top_K]
        video_feature = numpy.load("./output/PGM_feature/" + video_name + ".npy")
        video_feature = video_feature[:self.top_K, :]
        video_feature = torch.Tensor(video_feature)

        if self.mode == "train":
            video_match_iou = torch.Tensor(pdf.match_iou.values[:])
            return video_feature, video_match_iou
        else:
            video_xmin = pdf.xmin.values[:]
            video_xmax = pdf.xmax.values[:]
            video_xmin_score = pdf.xmin_score.values[:]
            video_xmax_score = pdf.xmax_score.values[:]
            return video_feature, video_xmin, video_xmax, video_xmin_score, video_xmax_score
