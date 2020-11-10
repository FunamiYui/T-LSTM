import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='neural model for factuality')
    parser.add_argument('--bert_model', type=str, default='./bert-base-uncased', help='path to the bert-base-uncased model')
    parser.add_argument('--train_data_path', type=str,
                        default='./unified/meantime/train.conll',
                        help='path to the training file')
    parser.add_argument('--dev_data_path', type=str,
                        default='./unified/meantime/dev.conll',
                        help='path to the development file')
    parser.add_argument('--test_data_path', type=str,
                        default='./unified/meantime/test.conll',
                        help='path to the testing file')

    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--train_batch_size', type=int, default=32, help='size of the training batches')
    parser.add_argument('--dev_batch_size', type=int, default=64, help='size of the development batches')
    parser.add_argument('--test_batch_size', type=int, default=64, help='size of the testing batches')

    parser.add_argument('--output_path', type=str,
                        default='./record/output_meantime.txt',
                        help='path to the output file')
    parser.add_argument('--inter_path', type=str,
                        default='./record/inter_meantime.txt',
                        help='path to the intermediate file')
    parser.add_argument('--model_path', type=str,
                        default='./checkpoint/baseline_meantime.pt',
                        help='path to the model')

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='coefficient of weight decay')

    parser.add_argument('--p_Asem', type=float, default=0.6,
                        help='trade off parameter labmda between the semantic and syntactic structures')
    parser.add_argument('--gcn_dropout', type=float, default=0.1,
                        help='dropout of gcn')

    args = parser.parse_args()
    for arg in vars(args):
        print('{}={}'.format(arg.upper(), getattr(args, arg)))
    print('')

    return args


args = parse_args()