import json
import torch
import random
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score
from Dataset import TextDataset
from Model import Classification
from Model import Conv1dRNN

""" configuration json을 읽어들이는 class """

random_seed = 42

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)



class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


def select_optimizer(model, args):
    lr = args['lr']
    beta1 = args['beta1']
    beta2 = args['beta2']
    eps = args['eps']
    weight_decay = args['weight_decay']
    amsgrad = args['amsgrad']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2),
                                 eps=eps, weight_decay=weight_decay,
                                 amsgrad=amsgrad)

    return optimizer


def select_scheduler(optimizer, config):
    T_max = config['T_max']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    return scheduler


def collate_fn(inputs):
    enc_inputs, dec_inputs, labels = list(zip(*inputs))
    enc_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True, padding_value=0)
    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0)

    batch = [
        enc_inputs,
        dec_inputs,
        torch.stack(labels, dim=0)
    ]
    return batch


def train(model, loader, criterion, optimizer, config):
    model.cuda()
    model.train()
    train_losses = []

    targets = []
    outputs = []
    train_cnt = 0
    train_corr = 0
    for idx, data in enumerate(loader):
        enc_inputs, dec_inputs, labels = data
        enc_inputs, dec_inputs, labels = enc_inputs.cuda(), dec_inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        if config.model == "conv1drnn":
            y_pred = model(enc_inputs)
        else:
            output = model(enc_inputs, dec_inputs)
            y_pred = output[0]

        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.detach())
        train_cnt += len(labels)

        train_corr += (labels == torch.argmax(y_pred, 1)).sum()
        targets += torch.argmax(y_pred, 1).tolist()
        outputs += labels.tolist()
    return model, sum(train_losses) / (idx + 1), f1_score(outputs, targets, average='macro'), train_corr / train_cnt


def evaluate(model, loader, criterion, config):
    model.cuda()
    model.eval()
    eval_losses = []

    targets = []
    outputs = []
    eval_cnt = 0
    eval_corr = 0
    for idx, data in enumerate(loader):
        enc_inputs, dec_inputs, labels = data
        enc_inputs, dec_inputs, labels = enc_inputs.cuda(), dec_inputs.cuda(), labels.cuda()

        if config.model == "conv1drnn":
            y_pred = model(enc_inputs)
        else:
            output = model(enc_inputs, dec_inputs)
            y_pred = output[0]

        loss = criterion(y_pred, labels)

        eval_losses.append(loss.detach())
        eval_cnt += len(labels)

        eval_corr += (labels == torch.argmax(y_pred, 1)).sum()
        targets += torch.argmax(y_pred, 1).tolist()
        outputs += labels.tolist()
    return model, sum(eval_losses) / (idx + 1), f1_score(outputs, targets, average='macro'), eval_corr / eval_cnt


def run(vocab_size=8000, verbose=True, early_stopping=True, separation=True):
    args = json.load(open("config.json"))
    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(args)
    print('Preparing Train.............')

    ### DATASET:
    DataSet = TextDataset(vocab_size=vocab_size, separation=separation)
    args['n_enc_vocab'] = vocab_size+7
    args['n_dec_vocab'] = vocab_size+7
    config.n_enc_vocab = vocab_size+7
    config.n_dec_vocab = vocab_size+7

    ### DATALOADER
    ratio = [int(len(DataSet) * args['train_ratio']), len(DataSet) - int(len(DataSet) * args['train_ratio'])]
    train_set, val_set = torch.utils.data.random_split(DataSet, ratio)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'],
                                               shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args['batch_size'],
                                             num_workers=2, collate_fn=collate_fn)

    ### MODEL
    if config.model == 'conv1drnn':
        model = Conv1dRNN(512, 3, config.dropout, config.n_dec_vocab, config.d_hidn, padding=1)
    else:
        model = Classification(config)

    ## CRITERION & OPTIMIZER & SECHEDULER
    criterion = nn.CrossEntropyLoss()
    optimizer = select_optimizer(model, args['Optim'])
    scheduler = select_scheduler(optimizer, args['Scheduler'])

    print('Training Start..............')
    history = {'train_losses': [], 'train_f1s': [], 'train_accs': [], 'eval_losses': [], 'eval_f1s': [],
               'eval_accs': []}
    ### RUN
    for i in range(args['Scheduler']['T_max']):
        model, train_loss, train_f1, train_acc = train(model, loader=train_loader, criterion=criterion,
                                                       optimizer=optimizer, config=config)
        model, eval_loss, eval_f1, eval_acc = evaluate(model, loader=val_loader, criterion=criterion, config=config)
        scheduler.step()

        history['train_losses'].append(train_loss)
        history['train_f1s'].append(train_f1)
        history['train_accs'].append(train_acc)
        history['eval_losses'].append(eval_loss)
        history['eval_f1s'].append(eval_f1)
        history['eval_accs'].append(eval_acc)
        if verbose:
            print(
                f'epoch {i + 1} train : train_loss = {train_loss:.4f}, train_f1 = {train_f1:.4f}, train_acc = {train_acc:.4f}')
            print(
                f'epoch {i + 1} validation : val_loss = {eval_loss:.4f}, val_f1 = {eval_f1:.4f}, val_acc = {eval_acc:.4f}')
            print('-' * 50)
        if early_stopping:
            current_eval_loss = eval_loss
            minimum_eval_loss = min(history['eval_losses'])
            if current_eval_loss > minimum_eval_loss:
                patience += 1
                if patience > 10:
                    print(f'early stop at best epoch {best_epoch}, valloss = {minimum_eval_loss}')
                    break
            else:
                best_model = model
                best_epoch = i
                patience = 0
    return best_model, history


if __name__ == '__main__':
    model, history = run()
