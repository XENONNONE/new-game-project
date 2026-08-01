import os
import pickle
import math
import shutil
import numpy as np
import lmdb as lmdb
import textgrid as tg
import pandas as pd
import torch
import glob
import json
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
#import pyarrow
import pickle
import librosa
import smplx
import glob

from .build_vocab import Vocab
from .utils.audio_features import Wav2Vec2Model
from .data_tools import joints_list
from .utils import rotation_conversions as rc
from .utils import other_tools


class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        self.args = args
        self.loader_type = loader_type

        self.rank = 0
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list = joints_list[self.args.tar_joints]
        if 'smplx' in self.args.pose_rep:
            self.joint_mask = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            self.joints = len(list(self.tar_joint_list.keys()))  
            for joint_name in self.tar_joint_list:
                self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        else:
            self.joints = len(list(self.ori_joint_list.keys()))+1
            self.joint_mask = np.zeros(self.joints*3)
            for joint_name in self.tar_joint_list:
                if joint_name == "Hips":
                    self.joint_mask[3:6] = 1
                else:
                    self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        # select trainable joints
        
        split_rule = pd.read_csv(args.data_path+"train_test_split.csv")
        self.selected_file = split_rule.loc[(split_rule['type'] == loader_type) & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
        if args.additional_data and loader_type == 'train':
            split_b = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            #self.selected_file = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            self.selected_file = pd.concat([self.selected_file, split_b])
        if self.selected_file.empty:
            logger.warning(f"{loader_type} is empty for speaker {self.args.training_speakers}, use train set 0-8 instead")
            self.selected_file = split_rule.loc[(split_rule['type'] == 'train') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            self.selected_file = self.selected_file.iloc[0:8]
        self.data_dir = args.data_path
        self.beatx_during_time = 0
        
        if loader_type == "test": 
            self.args.multi_length_training = [1.0]
        self.max_length = int(args.pose_length * self.args.multi_length_training[-1])
        self.max_audio_pre_len = math.floor(args.pose_length / args.pose_fps * self.args.audio_sr)
        if self.max_audio_pre_len > self.args.test_length*self.args.audio_sr: 
            self.max_audio_pre_len = self.args.test_length*self.args.audio_sr
        preloaded_dir = self.args.root_path + self.args.cache_path + loader_type + f"/{args.pose_rep}_cache"      

            
        if build_cache and self.rank == 0:
            self.build_cache(preloaded_dir)
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"] 
            
        self.norm = True
        self.mean = np.load('./mean_std/beatx_2_330_mean.npy')
        self.std = np.load('./mean_std/beatx_2_330_std.npy')
        
        self.trans_mean = np.load('./mean_std/beatx_2_trans_mean.npy')
        self.trans_std = np.load('./mean_std/beatx_2_trans_std.npy')
    
    def build_cache(self, preloaded_dir):
        logger.info(f"Audio bit rate: {self.args.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        logger.info("Creating the dataset cache...")
        if self.args.new_cache:
            if os.path.exists(preloaded_dir):
                shutil.rmtree(preloaded_dir)
        if os.path.exists(preloaded_dir):
            logger.info("Found the cache {}".format(preloaded_dir))
        elif self.loader_type == "test":
            self.cache_generation(
                preloaded_dir, True, 
                0, 0,
                is_test=True)
        else:
            self.cache_generation(
                preloaded_dir, self.args.disable_filtering, 
                self.args.clean_first_seconds, self.args.clean_final_seconds,
                is_test=False)
        logger.info(f"BEATX during time is {self.beatx_during_time}s !")
        
        
    def __len__(self):
        return self.n_samples
    

    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds,  clean_final_seconds, is_test=False):
        self.n_out_samples = 0
        # create db for samples
        if not os.path.exists(out_lmdb_dir): os.makedirs(out_lmdb_dir)
        dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size= int(1024 ** 3 * 50))# 50G
        n_filtered_out = defaultdict(int)
    
        for index, file_name in self.selected_file.iterrows():
            f_name = file_name["id"]
            ext = ".npz" if "smplx" in self.args.pose_rep else ".bvh"
            pose_file = self.data_dir + self.args.pose_rep + "/" + f_name + ext
            pose_each_file = []
            trans_each_file = []
            trans_v_each_file = []
            shape_each_file = []
            audio_each_file = []
            facial_each_file = []
            word_each_file = []
            emo_each_file = []
            sem_each_file = []
            vid_each_file = []
            id_pose = f_name #1_wayne_0_1_1
            
            logger.info(colored(f"# ---- Building cache for Pose   {id_pose} ---- #", "blue"))
            if "smplx" in self.args.pose_rep:
                pose_data = np.load(pose_file, allow_pickle=True)
                assert 30%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 30'
                stride = int(30/self.args.pose_fps)
                pose_each_file = pose_data["poses"][::stride] * self.joint_mask
                pose_each_file = pose_each_file[:, self.joint_mask.astype(bool)]
                
                self.beatx_during_time += pose_each_file.shape[0]/30
                trans_each_file = pose_data["trans"][::stride]
                trans_each_file[:,0] = trans_each_file[:,0] - trans_each_file[0,0]
                trans_each_file[:,2] = trans_each_file[:,2] - trans_each_file[0,2]
                trans_v_each_file = np.zeros_like(trans_each_file)
                trans_v_each_file[1:,0] = trans_each_file[1:,0] - trans_each_file[:-1,0]
                trans_v_each_file[0,0] = trans_v_each_file[1,0]
                trans_v_each_file[1:,2] = trans_each_file[1:,2] - trans_each_file[:-1,2]
                trans_v_each_file[0,2] = trans_v_each_file[1,2]
                trans_v_each_file[:,1] = trans_each_file[:,1]
                
                
                shape_each_file = np.repeat(pose_data["betas"].reshape(1, 300), pose_each_file.shape[0], axis=0)
                if self.args.facial_rep is not None:
                    logger.info(f"# ---- Building cache for Facial {id_pose} and Pose {id_pose} ---- #")
                    facial_each_file = pose_data["expressions"][::stride]
                    if self.args.facial_norm: 
                        facial_each_file = (facial_each_file - self.mean_facial) / self.std_facial
                    
            if self.args.id_rep is not None:
                vid_each_file = np.repeat(np.array(int(f_name.split("_")[0])-1).reshape(1, 1), pose_each_file.shape[0], axis=0)
      
            filtered_result = self._sample_from_clip(
                dst_lmdb_env,
                pose_each_file, trans_each_file,trans_v_each_file, shape_each_file, facial_each_file,
                vid_each_file,
                disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
                ) 
            for type in filtered_result.keys():
                n_filtered_out[type] += filtered_result[type]
                        
        with dst_lmdb_env.begin() as txn:
            logger.info(colored(f"no. of samples: {txn.stat()['entries']}", "cyan"))
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                logger.info("{}: {}".format(type, n_filtered))
                n_total_filtered += n_filtered
            logger.info(colored("no. of excluded samples: {} ({:.1f}%)".format(
                n_total_filtered, 100 * n_total_filtered / (txn.stat()["entries"] + n_total_filtered)), "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()
    
    def _sample_from_clip(
        self, dst_lmdb_env, pose_each_file, trans_each_file, trans_v_each_file, shape_each_file, facial_each_file,
        vid_each_file,
        disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
        ):
        """
        for data cleaning, we ignore the data for first and final n s
        for test, we return all data 
        """
        
        round_seconds_skeleton = pose_each_file.shape[0] // self.args.pose_fps  # assume 1500 frames / 15 fps = 100 s
        #print(round_seconds_skeleton)
 
        clip_s_t, clip_e_t = clean_first_seconds, round_seconds_skeleton - clean_final_seconds # assume [10, 90]s
        clip_s_f_audio, clip_e_f_audio = self.args.audio_fps * clip_s_t, clip_e_t * self.args.audio_fps # [160,000,90*160,000]
        clip_s_f_pose, clip_e_f_pose = clip_s_t * self.args.pose_fps, clip_e_t * self.args.pose_fps # [150,90*15]
        
        
        for ratio in self.args.multi_length_training:
            if is_test:# stride = length for test
                cut_length = clip_e_f_pose - clip_s_f_pose
                self.args.stride = cut_length
                self.max_length = cut_length
            else:
                self.args.stride = int(ratio*self.ori_stride)
                cut_length = int(self.ori_length*ratio)
                
            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / self.args.stride) + 1
            logger.info(f"pose from frame {clip_s_f_pose} to {clip_e_f_pose}, length {cut_length}")
            logger.info(f"{num_subdivision} clips is expected with stride {self.args.stride}")
            

            n_filtered_out = defaultdict(int)
            sample_pose_list = []
            sample_face_list = []
            sample_shape_list = []
            sample_vid_list = []
            sample_trans_list = []
            sample_trans_v_list = []
           
            for i in range(num_subdivision): # cut into around 2s chip, (self npose)
                start_idx = clip_s_f_pose + i * self.args.stride
                fin_idx = start_idx + cut_length 
                sample_pose = pose_each_file[start_idx:fin_idx]
                sample_trans = trans_each_file[start_idx:fin_idx]
                sample_trans_v = trans_v_each_file[start_idx:fin_idx]
                sample_shape = shape_each_file[start_idx:fin_idx]
                sample_face = facial_each_file[start_idx:fin_idx]
                # print(sample_pose.shape)
                sample_vid = vid_each_file[start_idx:fin_idx] if self.args.id_rep is not None else np.array([-1])
                
                if sample_pose.any() != None:
                    sample_pose_list.append(sample_pose)

                    sample_shape_list.append(sample_shape)

                    sample_vid_list.append(sample_vid)
                    sample_face_list.append(sample_face)


                    sample_trans_list.append(sample_trans)
                    sample_trans_v_list.append(sample_trans_v)

            if len(sample_pose_list) > 0:
                with dst_lmdb_env.begin(write=True) as txn:
                    for pose, shape, face, vid, trans,trans_v in zip(
                        sample_pose_list,
                        sample_shape_list,
                        sample_face_list,
                        sample_vid_list,
                        sample_trans_list,
                        sample_trans_v_list,
                        ):
                        k = "{:005}".format(self.n_out_samples).encode("ascii")
                        v = [pose , shape, face, vid, trans,trans_v]
                        v = pickle.dumps(v,5)
                        txn.put(k, v)
                        self.n_out_samples += 1
        return n_filtered_out

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pickle.loads(sample)
            tar_pose,  in_shape, tar_face, vid, trans,trans_v = sample
            tar_pose = torch.from_numpy(tar_pose).float()
            tar_face = torch.from_numpy(tar_face).float()
            tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(-1, 55, 3))
            tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(-1, 55*6)
            
            if self.norm:
                tar_pose = (tar_pose - self.mean) / self.std
                trans_v = (trans_v-self.trans_mean)/self.trans_std
            
            if self.loader_type == "test":
                tar_pose = tar_pose.float()
                trans = torch.from_numpy(trans).float()
                trans_v = torch.from_numpy(trans_v).float()
                vid = torch.from_numpy(vid).float()
                in_shape = torch.from_numpy(in_shape).float()
                tar_pose = torch.cat([tar_pose, trans_v], dim=1)
                tar_pose = torch.cat([tar_pose, tar_face], dim=1)
            else:
                in_shape = torch.from_numpy(in_shape).reshape((in_shape.shape[0], -1)).float()
                trans = torch.from_numpy(trans).reshape((trans.shape[0], -1)).float()
                trans_v = torch.from_numpy(trans_v).reshape((trans_v.shape[0], -1)).float()
                vid = torch.from_numpy(vid).reshape((vid.shape[0], -1)).float()
                tar_pose = tar_pose.reshape((tar_pose.shape[0], -1)).float()
                tar_pose = torch.cat([tar_pose, trans_v], dim=1)
                tar_pose = torch.cat([tar_pose, tar_face], dim=1)
            return tar_pose