from torch.utils.data import Dataset
import kaldi_io
import numpy as np
from lib.Data_show import Data_show




class Phone_cla_Dataset(Dataset):
    """Face Landmarks dataset."""
    

    maxClassNum =  -1
    class_trans_vector = None

    def __init__(self,phone_label=None, feats=None, transform=None):
        """
        Args:
            phone_label (dict): utt to frame label.
            feats (dict): utt to frame features.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if Phone_cla_Dataset.class_trans_vector == None:
            Phone_cla_Dataset.class_trans_vector = np.vectorize(Phone_cla_Dataset.class_trans)
            Phone_cla_Dataset.maxClassNum = max(list(Data_show.phone2class.values())) + 1
        
        if phone_label == None or feats == None:
            self.phone_label = { u:d for u,d in kaldi_io.read_vec_int_ark("feats/ali.1.ph") }
            self.feats = { u:d for u,d in kaldi_io.read_mat_scp("feats/feats.scp") }
        else:
            self.phone_label = phone_label
            self.feats = feats
        
        self.feats_list = []
        self.phone_label_list = []

        self.transform = transform
        
        for utt, feat in feats.items():
            if utt in phone_label:
                self.feats_list.append(feat)
                a=np.zeros(feat.shape[0], int)
                for i in range(a.shape[0]):
                    a[i]=phone_label[utt][(i)//3]
                self.phone_label_list.append(Phone_cla_Dataset.class_trans_vector(a))

        self.feats_nd = np.concatenate(tuple(self.feats_list))
        self.phone_label_nd = np.concatenate(tuple(self.phone_label_list))
           

    def __len__(self):
        return len(self.phone_label_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = [self.feats_list[idx], self.phone_label_list[idx]]

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def class_trans(x):
        if x in Data_show.phone2class:
            result = Data_show.phone2class[x]
        else:
            result = Phone_cla_Dataset.maxClassNum

        return result

        