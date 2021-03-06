class Data_show:
    phone2class={}
    phones2txt = {}
    
    def __init__(self, phone2class ={1:0, 3:0,129:1, 63:2, 61:3, 27:4, 128:5, 64:6, 92:7, 69:8}, phones_path="phones.txt"):
        
        Data_show.phone2class = phone2class
        with open(phones_path) as f:
            for line in f:
                v = line.strip().split()
                Data_show.phones2txt[int(v[1])] = v[0]

    def show_softmax(self, pred_label_with_softmax):
        
        title = ""
        for i in Data_show.phone2class.keys():
            title += "\t" + Data_show.phones2txt[i]
        title += "\tother\n"
        
        content = ""
        for i in range(0,pred_label_with_softmax.shape[0]):
            content += str(i) + ":\t"
            for j in range(0,pred_label_with_softmax.shape[1]):
                content += '%.2f\t' %pred_label_with_softmax[i][j]
            content += "\n"
     
        return title, content
    
    
class Data_show_word:
    class2word=[]
    
    def __init__(self, class2word =["other", "你","好","米","雅"]):
        
        Data_show_word.class2word = class2word


    def show_softmax(self, pred_label_with_softmax):
        
        title = ""
        for i in range(len(Data_show_word.class2word)):
            title += "\t" + Data_show_word.class2word[i]
        title += "\n"
        
        content = ""
        for i in range(0,pred_label_with_softmax.shape[0]):
            content += str(i) + ":\t"
            for j in range(0,pred_label_with_softmax.shape[1]):
                content += '%.2f\t' %pred_label_with_softmax[i][j]
            content += "\n"
     
        return title, content
    
