class Data_show:
    phone2class={1:0, 3:0,129:1, 63:2, 61:3, 27:4, 128:5, 64:6, 92:7, 69:8}
    phones2txt = {}
    
    def __init__(self, phone2class ={1:0, 3:0,129:1, 63:2, 61:3, 27:4, 128:5, 64:6, 92:7, 69:8}, phones_path="phones.txt"):
        
        self.phone2class = phone2class
        with open(phones_path) as f:
            for line in f:
                v = line.strip().split()
                self.phones2txt[int(v[1])] = v[0]

    def show_softmax(self, pred_label_with_softmax):
        
        title = ""
        for i in self.phone2class.keys():
            title += "\t" + self.phones2txt[i]
        title += "\tother\n"
        
        content = ""
        for i in range(0,pred_label_with_softmax.shape[0]):
            content += str(i) + ":\t"
            for j in range(0,pred_label_with_softmax.shape[1]):
                content += '%.2f\t' %pred_label_with_softmax[i][j]
            content += "\n"
     
        return title, content
    
