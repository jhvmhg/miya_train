from torch.utils.data import Dataset
import kaldi_io

class Phone_cla_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,phone_label=None, feats=None, transform=None):
        """
        Args:
            phone_label (dict): utt to frame label.
            feats (dict): utt to frame features.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if phone_label == None or feats == None:
            self.phone_label = { u:d for u,d in kaldi_io.read_vec_int_ark("feats/ali.1.ph") }
            self.feats = { u:d for u,d in kaldi_io.read_mat_scp("feats/feats.scp") }
        else:
            self.phone_label = phone_label
            self.feats = feats
        
        self.feats_list = []
        self.phone_label_list = []

        self.transform = transform
        
        for utt, phones in phone_label.items():
            self.feats_list.append(feats[utt])
            a=np.zeros(feats[utt].shape[0], int)
            for i in range(a.shape[0]):
                a[i]=phone_label[utt][(i)//3]
            self.phone_label_list.append(function_vector(a))
#             self.phone_label_list = np.concatenate(( self.phone_label_list,self.phone_label[utt]))
            
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
    
        