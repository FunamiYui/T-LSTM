import os

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange

from data_utils import DataLoaderBert
from Model import BaselineBert


def tokenizer(text):
    return [tok for tok in text]


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def trainer_train(epochs):
    best_r = 0.0
    for epoch in trange(epochs):
        model.train()
        temp_loss = 0.0
        count = 0

        for trigger_emb, eep in train_iter:
            trigger_emb = trigger_emb.to(device)
            eep = eep.squeeze().to(device)

            optimizer.zero_grad()

            assert trigger_emb.shape[1] == 768, "BERT hidden dim wrong"
            out = model(trigger_emb)

            loss = F.smooth_l1_loss(out, eep)
            loss.backward()
            optimizer.step()

            temp_loss += loss.item()
            count += 1

        print('[epoch %d, %d] loss: %f' % (epoch, count, temp_loss / count))

        _, test_r, _ = trainer_test(1)
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


def trainer_test(epochs):
    model.eval()
    loss_list = 0.0
    for epoch in trange(epochs):
        temp_loss = 0.0
        count = 0
        eval_history_out = []
        eval_history_label = []

        for trigger_emb, eep in test_iter:
            trigger_emb = trigger_emb.to(device)
            eep = eep.squeeze().to(device)

            out = model(trigger_emb)

            loss = F.smooth_l1_loss(out, eep)
            temp_loss += loss.item()
            count += 1

            eval_history_out = eval_history_out + out.cpu().detach().numpy().tolist()
            eval_history_label = eval_history_label + eep.cpu().detach().numpy().tolist()

        loss_list += temp_loss / count
        r = pearsonr(eval_history_out, eval_history_label)
        mae = mean_absolute_error(eval_history_out, eval_history_label)
        return loss_list / epochs, r, mae


if __name__ == '__main__':
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.manual_seed_all(0)

    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # writer = SummaryWriter('./tensorboard/baseline/meantime')

    epoch = 300
    train_batch_size = 32
    dev_batch_size = 128
    test_batch_size = 128

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
                           shuffle=False, drop_last=True)

    model_path = "./checkpoint/baseline_meantime.pt"
    model = BaselineBert()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=5e-4)

    print("Start training...")
    trainer_train(epoch)

    print("Start testing...")
    checkpoint = torch.load(model_path)
    model.load_state_dict = (checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(checkpoint['epoch'], checkpoint['correlation'])

    test_loss, test_r, test_mae = trainer_test(1)
    print("test_loss: ", test_loss, "test_r: ", test_r, "test_mae: ", test_mae)
