class Decoder():
#     pred_label_aishell_with_softmax = torch.nn.Softmax()(pred_label_aishell)
    def __init__(self,phone2class):
        self.s = -0.2
        self.m = -0.2
        
        self.phone2class = phone2class

    def decode(self, pred_label_aishell_with_softmax):
 

        F = np.zeros((len(Data_show.phone2class)+1, pred_label_aishell_with_softmax.shape[0]))
        for t in range(1,pred_label_aishell_with_softmax.shape[0]):
            for i in range(2,len(Data_show.phone2class)+1):
                F[i,t] = pred_label_aishell_with_softmax[t-1,i-1] + max(self.s+F[i,t-1], self.m+F[i-1,t-1])
                
        return F
    
    def show_result(self, F):
        
        title = ""
        for i in self.phone2class.keys():
            title += "\t" + Data_show.phones2txt[i]
        title += "\tother\n"
        
        content = ""
        for i in range(1,F.shape[1]):
            content += str(i) + ":\t"
            for j in range(1,F.shape[0]):
                content +='%.2f\t' %F[j][i]
            content += "\n"
            
        return title, content