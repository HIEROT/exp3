import numpy as np
import torch

from model import GGNN


class Detect(object):
    """
        dir_name: Folder or image_file
    """

    def __init__(self, weights):
        super(Detect,  self).__init__()
        self.weights = weights
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else 'cpu')
        print('Load pretrained Model')
        checkpoint = torch.load(
                self.weights, map_location=lambda storage, loc: storage)
        self.args = checkpoint['options']

        self.model = GGNN(self.args)

        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict)
        self.model = self.model.cuda()
        self.model.eval()

    def detect(self, dataset=None, savefile=None):
        predlist =[]
        for adj_matrix, annotation, _ in dataset:
            adj_matrix = torch.tensor(adj_matrix).to(self.device)
            annotation = torch.tensor(annotation).to(self.device)
            padding = torch.zeros(len(annotation), self.args.n_node, self.args.state_dim - self.args.annotation_dim)
            init_input = torch.cat((annotation, padding), 2).to(self.device)
            classification = self.model(init_input, annotation, adj_matrix)
            pred = classification.max(1, keepdim=True)[1]
            predlist.append(pred)
        if savefile:
            np.save(savefile, np.array(predlist))
        return predlist
