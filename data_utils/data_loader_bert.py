import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import args
from data_utils import LoadData
from Model import BaselineBert
from tqdm import trange


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
        super(DataLoaderBert, self).__init__()

        loaddata = LoadData(train_path, dev_path, test_path)
        a = loaddata.conllu_counter[dataset]
        counter = loaddata.counter_process(a)

        tokenizer = BertTokenizer.from_pretrained(args.bert_model)

        max_length = 0
        for i in range(len(counter)):
            if len(counter[i].sentence) > max_length:
                max_length = len(counter[i].sentence)
        print("max_length", max_length)
        max_length += 2  # [CLS], [SEP]

        for i in trange(len(counter)):
            assert len(counter[i].trigger) == 1, "not one trigger per sentence"
            counter[i].sentence_emb = torch.tensor(tokenizer.encode(counter[i].sentence, padding='max_length',
                                                                    max_length=max_length))  # [seq_len]
            counter[i].mask = torch.zeros(max_length)
            counter[i].mask[:len(counter[i].sentence)+2] = 1
            counter[i].adj_matrix = counter[i].trans_data(max_length - 2)
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
        out = model(sentence_emb, trigger_index, mask)
        # accu = F.l1_loss(out, eep)
        # print(accu)
