import numpy as np
import os
import re
import json
import audiofeatureextraction as aft
import torch


def FolderAnalysis(root, files, classflag):
    datadict = {'adj_matrix': None, 'annotation': None, 'target': classflag}
    vertex_idx = 0
    bodymovements = []
    emotions = np.zeros((30, 7))
    lastframepos = []
    numarray = []
    for i in range(30):
        if "%d_keypoints.json" % i in files:
            with open(os.path.join(root, "%d_keypoints.json" % i), 'r') as load_f:
                load_array = json.load(load_f)['people']
            num_people = len(load_array)
            current_pos = np.array([[load_array[j]['pose_keypoints_2d'][k * 3:k * 3 + 2]
                                     for k in [0, 15, 16, 3, 4, 6, 7]]
                                    for j in range(num_people)])
            difference = current_pos
            if len(lastframepos) == num_people:
                # 这里假定了复数个人前后两帧都在场的情况下，人的排列顺序与前一帧相同
                difference = current_pos - lastframepos
            # 对头部差求均值，余下不变
            temp = np.reshape(np.concatenate((np.average(difference[:, :3], 1, current_pos[:, 0:3] > 0)
                                                            [:, np.newaxis, :], difference[:, 3:]),
                                                           1), (-1, 10))
            for ele in temp:
                bodymovements.append(ele)
            vertex_idx += num_people
            numarray.append(num_people)
        if "%d_emotions.npy" % i in files:
            emotions[i, :] = np.load(os.path.join(root, "%d_emotions.npy" % i))
    # 处理音频
    audiofeatures = aft.ExtractFromAudio(os.path.join(root, "audio.wav"), 30)
    bodymovements = np.array(bodymovements)
    adj_matrix = np.zeros((vertex_idx, vertex_idx * 2))
    annotation = np.zeros((vertex_idx, np.shape(emotions)[1] + np.shape(bodymovements)[1] + np.shape(audiofeatures)[1]))
    frame_idx = 0
    sum_vertex = numarray[0]
    for i in range(vertex_idx):
        if i >= sum_vertex:
            frame_idx += 1
            sum_vertex += numarray[frame_idx]
        # 生成顶点
        temp = np.squeeze(np.concatenate((emotions[frame_idx, :], bodymovements[i, :],
                                                      audiofeatures[frame_idx, :])))
        annotation[i, :] = temp
        # 生成边
        if frame_idx:
            if numarray[frame_idx] == numarray[frame_idx - 1]:
                adj_matrix[i, i - numarray[frame_idx]] = 1
                adj_matrix[i - numarray[frame_idx], i + vertex_idx] = 1
            else:
                adj_matrix[i, (sum_vertex - numarray[frame_idx]
                               - numarray[frame_idx - 1]):(sum_vertex - numarray[frame_idx])] = 1
                adj_matrix[(sum_vertex - numarray[frame_idx]
                            - numarray[frame_idx - 1]):(sum_vertex - numarray[frame_idx]), i + vertex_idx] = 1
    datadict['annotation'] = annotation
    datadict['adj_matrix'] = adj_matrix
    return datadict


class DataDetect:
    def __init__(self, dataset_folder_path):
        self.datalist = []
        classflagdict = {'positive': 1, 'negative': 0, 'test': None}
        classflag = None
        for root, dirs, files in os.walk(top=dataset_folder_path, topdown=True):
            if re.match(r'[0-9]*', os.path.split(root)[1]):
                # 是数据存在的目录
                datadict = FolderAnalysis(root, files, classflag)
                self.datalist.append(datadict)
            elif re.match(r'(positive)|(negative)|(test)', os.path.split(root)[1]):
                classflag = classflagdict[os.path.split(root)[1]]
        max_num_nodes = 0
        for i in self.datalist:
            if np.shape(i['adj_matrix'])[0] > max_num_nodes:
                max_num_nodes = np.shape(i['adj_matrix'])[0]
        for i in self.datalist:
            new_adj = np.zeros((max_num_nodes, max_num_nodes * 2))
            num_nodes = np.shape(i['adj_matrix'])[0]
            new_adj[:num_nodes, :num_nodes] = i['adj_matrix'][:, :num_nodes]
            new_adj[:num_nodes, max_num_nodes:max_num_nodes + num_nodes] = i['adj_matrix'][:, num_nodes:]
            i['adj_matrix'] = new_adj
            i['annotation'] = np.concatenate((i['annotation'], np.zeros((max_num_nodes - num_nodes,
                                                            np.shape(i['annotation'])[1]))), axis=0)
        self.max_num_nodes = max_num_nodes

    def __getitem__(self, item):
        return self.datalist[item]['adj_matrix'], self.datalist[item]['annotation'], self.datalist[item]['target']

    def __len__(self):
        return len(self.datalist)


if __name__ == '__main__':
    for root, dirs, files in os.walk(top='../../../dataset/train/positive/2'):
        accdict = FolderAnalysis(root, files, 1)
        print(accdict)