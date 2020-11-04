import os

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch.nn.functional as F

from data_utils import DataLoaderBert
from Model import BaselineBert, GraphBaseline


def tokenizer(text):
    return [tok for tok in text]


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def trainer_train(epochs):
    best_r = 0.0
    for epoch in range(epochs):
        model.train()
        temp_loss = 0.0
        count = 0

        for sentence_emb, mask, adj_matrix, eep, trigger_index, trigger in train_iter:
            sentence_emb = sentence_emb.to(device)  # [batch, seq_len, bert_dim]
            mask = mask.to(device)  # [batch, seq_len]
            adj_matrix = adj_matrix.to(device).to_dense()  # [batch, seq_len, seq_len]
            eep = eep.to(device)  # [batch]
            trigger_index = trigger_index.to(device)  # [batch]

            optimizer.zero_grad()

            out = model(sentence_emb)

            loss = F.smooth_l1_loss(out, eep)
            loss.backward()
            optimizer.step()

            temp_loss += loss.item()
            count += 1

        print('[epoch %d, %d] loss: %f' % (epoch, count, temp_loss / count))

        _, test_r, _ = trainer_test()
        if test_r[0] > best_r:
            best_r = test_r[0]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'correlation': test_r[0],
            }, model_path)



'''
def trainer_dev(epoch):
    loss_list = 0.0
    for cycle in trange(epoch):
        temp_loss = 0.0
        count = 0
        model.eval()

        for trigger_emb, eep in dev_iter:
            trigger_emb = trigger_emb.to(device)
            eep = eep.squeeze().to(device)

            # optimizer.zero_grad()
            out = model(trigger_emb)
            # loss = F.smooth_l1_loss(out, eep)
            accu = F.l1_loss(out, eep)
            # loss.backward()
            # optimizer.step()
            temp_loss += accu.item()
            count += 1
        # print("dev loss:",(temp_loss / count))
        loss_list += (temp_loss / count)
        return loss_list / epoch
'''


def trainer_test(epochs=1, wr=False):
    f = open(filename, 'w')

    model.eval()
    loss_list = 0.0
    for epoch in range(epochs):
        temp_loss = 0.0
        count = 0
        eval_history_out = []
        eval_history_label = []

        for sentence_emb, mask, adj_matrix, eep, trigger_index, trigger in test_iter:
            sentence_emb = sentence_emb.to(device)  # [batch, seq_len, bert_dim]
            mask = mask.to(device)  # [batch, seq_len]
            adj_matrix = adj_matrix.to(device).to_dense()  # [batch, seq_len, seq_len]
            eep = eep.to(device)  # [batch]
            trigger_index = trigger_index.to(device)  # [batch]

            out = model(sentence_emb)

            loss = F.smooth_l1_loss(out, eep)
            temp_loss += loss.item()
            count += 1

            eval_history_out = eval_history_out + out.cpu().detach().numpy().tolist()
            eval_history_label = eval_history_label + eep.cpu().detach().numpy().tolist()

            if wr:
                batch = eep.shape[0]
                for i in range(batch):
                    # trigger_index, trigger, truth, prediction
                    line = str(trigger_index[i].item()) + "\t" + trigger[i] + "\t" + str(eep[i].item()) + "\t" + str(out[i].item()) + "\n"
                    f.write(line)

        loss_list += temp_loss / count
        r = pearsonr(eval_history_out, eval_history_label)
        mae = mean_absolute_error(eval_history_out, eval_history_label)

        if wr:
            f.write("test_loss: " + str(loss_list / epochs) + "\n")
            f.write("test_r: " + str(r[0]) + "\n")
            f.write("test_mae: " + str(mae) + "\n")
        f.close()
        return loss_list / epochs, r, mae


if __name__ == '__main__':
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda")
    # writer = SummaryWriter('./tensorboard/baseline/meantime')

    epoch = 500
    train_batch_size = 32
    dev_batch_size = 128
    test_batch_size = 64

    print("Prepare data...")
    train_dataset = DataLoaderBert("./unified/meantime/train.conll", "./unified/meantime/dev.conll",
                                   "./unified/meantime/test.conll", 'train')
    '''
    dev_dataset = DataLoaderBert("./unified/meantime/train.conll", "./unified/meantime/dev.conll",
                    "./unified/meantime/test.conll", 'dev')
    '''
    test_dataset = DataLoaderBert("./unified/meantime/train.conll", "./unified/meantime/dev.conll",
                                  "./unified/meantime/test.conll", 'test')

    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=train_batch_size,
                            shuffle=True)
    '''
    dev_iter = DataLoader(dataset=dev_dataset,
                          batch_size=dev_batch_size,
                          shuffle=True, drop_last=True)
    '''

    test_iter = DataLoader(dataset=test_dataset,
                           batch_size=test_batch_size,
                           shuffle=False)

    filename = "./record/baseline_meantime2.txt"
    model_path = "./checkpoint/baseline_meantime2.pt"
    model = BaselineBert()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    print("Start training...")
    trainer_train(epoch)

    print("Start testing...")
    test_loss, test_r, test_mae = trainer_test()
    print("test_loss:", test_loss, "test_r:", test_r, "test_mae:", test_mae)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(checkpoint['epoch'], checkpoint['correlation'])

    test_loss, test_r, test_mae = trainer_test(wr=True)
    print("test_loss:", test_loss, "test_r:", test_r, "test_mae:", test_mae)
