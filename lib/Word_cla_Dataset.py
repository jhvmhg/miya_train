from torch.utils.data import Dataset
import kaldi_io
import numpy as np



class Word_cla_Dataset(Dataset):
    """Face Landmarks dataset."""
    


    def __init__(self,word_label=None, feats=None, transform=None):
        """
        Args:
            phone_label (dict): utt to frame label.
            feats (dict): utt to frame features.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        
        if word_label == None or feats == None:
#             self.word_label = { u:d for u,d in kaldi_io.read_vec_int_ark("feats/ali.1.ph") }
            self.feats = { u:d for u,d in kaldi_io.read_mat_scp("feats/feats.scp") }
        else:
            self.word_label = word_label
            self.feats = feats
        
        self.feats_list = []
        self.word_label_list = []

        self.transform = transform
        
        for utt, feat in feats.items():
            if utt in word_label:
                a=np.zeros(feat.shape[0]-29, int)

                for i in range(20,feat.shape[0]-9):
                    input_data=feat[i-20:i+10].reshape(1,-1)
                    if input_data.shape != (1,1200):
                        print(input_data.shape,"\t",i,utt)
                    self.feats_list.append(input_data)
                    a[i-20]=word_label[utt][(i)//3]
                        
                self.word_label_list.append(a)

       
        self.feats_nd = np.concatenate(tuple(self.feats_list))
        self.word_label_nd = np.concatenate(tuple(self.word_label_list))
           

    def __len__(self):
        return len(self.word_label_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = [self.feats_list[idx], self.word_label_list[idx]]

        if self.transform:
            sample = self.transform(sample)

        return sample
    


