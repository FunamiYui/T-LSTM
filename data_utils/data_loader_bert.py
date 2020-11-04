from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np
from data_utils import LoadData
from transformers import BertTokenizer, BertModel
from Model import BaselineBert


class DataLoaderBert(Dataset):
    """
        To predict the factuality v_t for the event
        referred to by a word w_t,
        use the contextualized embeddings
        in the last layer of the pre-trained BERT model
        as the input to a two-layer regression model.

        NOTE: assert w_t is the trigger, and there is only one trigger per sentence.
    """

    def __init__(self, train_path, dev_path, test_path, dataset):
        loaddata = LoadData(train_path, dev_path, test_path)
        a = loaddata.conllu_counter[dataset]
        counter = loaddata.counter_process(a)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")
        model.cuda()

        model.eval()

        max_len = 0
        for i in range(len(counter)):
            if len(counter[i].sentence) > max_len:
                max_len = len(counter[i].sentence)
        print("max_len", max_len)

        for i in trange(len(counter)):
            assert len(counter[i].trigger) == 1, "not one trigger per sentence"
            ids = torch.tensor(tokenizer.encode(counter[i].sentence)).unsqueeze(0).cuda()  # [batch=1, seq_len]
            
            with torch.no_grad():
                h = model(ids)[0][:, 1:-1, :].squeeze(0).cpu()  # [seq_len, bert_dim]
                # counter[i].trigger_emb = h[counter[i].trigger_index[0]]  # [bert_dim]
                seq_len, bert_dim = h.shape
                assert seq_len == len(counter[i].sentence), "after emb"
                counter[i].sentence_emb = torch.zeros([max_len, bert_dim])
                counter[i].sentence_emb[:seq_len] = h

                counter[i].mask = torch.zeros(max_len)
                counter[i].mask[:seq_len] = torch.ones(seq_len)

                counter[i].adj_matrix = counter[i].trans_data(max_len)
            counter[i].index = i
        self.data = counter
        self.len = len(self.data)
        print(dataset, self.len)

    def __getitem__(self, index):
        '''
        return self.data[index].sentence, self.data[index].sentence_emb, self.data[index].index, torch.tensor(
            self.data[index].trigger_index,dtype=torch.long), torch.tensor(self.data[index].eep)
        '''
        return self.data[index].sentence_emb, self.data[index].mask, self.data[index].adj_matrix, \
               torch.tensor(self.data[index].eep[0]), torch.tensor(self.data[index].trigger_index[0],
                                                                   dtype=torch.long), \
               self.data[index].trigger[0]

    def __len__(self):
        return self.len


if __name__ == "__main__":
    train_dataset = DataLoaderBert("../unified/meantime/train.conll", "../unified/meantime/dev.conll",
                                           "../unified/meantime/test.conll", 'train')

    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=32,
                            shuffle=False)

    model = BaselineBert()
    for sentence_emb, mask, adj_matrix, eep, trigger_index, trigger in train_iter:
        print(sentence_emb.shape)
        print(mask.shape)
        print(adj_matrix.shape)
        print(eep.shape)
        print(trigger_index.shape)
        out = model(sentence_emb)
        # accu = F.l1_loss(out, eep)
        # print(accu)
