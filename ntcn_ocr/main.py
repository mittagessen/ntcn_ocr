#!/usr/bin/env python3
import numpy as np

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from collections import defaultdict
from ntcn_ocr.model import ConvSeqNet, TorchSeqRecognizer
import click

from kraken.lib.train import EpochStopping
from kraken.lib.dataset import GroundTruthDataset, generate_input_transforms, InfiniteDataLoader, compute_error

@click.group()
def cli():
    pass

def collate_sequences(batch):
    """
    Sorts and pads sequences.
    """
    for x in batch:
        x['image'] = x['image'].squeeze()
    sorted_batch = sorted(batch, key=lambda x: x['image'].shape[1], reverse=True)
    seqs = [x['image'] for x in sorted_batch]
    seq_lens = torch.LongTensor([seq.shape[1] for seq in seqs])
    max_len = seqs[0].shape[1]
    seqs = torch.stack([F.pad(seq, pad=(0, max_len-seq.shape[1])) for seq in seqs])
    if isinstance(sorted_batch[0]['target'], str):
        labels = [x['target'] for x in sorted_batch]
    else:
        labels = torch.cat([x['target'] for x in sorted_batch]).long()
    label_lens = torch.LongTensor([len(x['target']) for x in sorted_batch])
    return {'image': seqs, 'target': labels, 'seq_lens': seq_lens, 'target_lens': label_lens}


@cli.command()
@click.option('-m', '--model', default=None, help='model name')
@click.option('-w', '--workers', default=0, help='number of workers loading training data')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.option('--valid-seq-len', default=320, help='part of the training sample used for back propagation')
@click.option('--seq-len', default=400, help='total training sample sequence length')
@click.option('--hidden', default=100, help='numer of hidden units per block')
@click.option('--layers', default=3, help='number of 3-layer blocks')
@click.option('--kernel', default=3, help='kernel size')
@click.option('-N', '--batch-size', default=128, help='batch size')
@click.option('-r', '--regularization', default='dropout2d', type=click.Choice(['dropout', 'dropout2d', 'batchnorm']))
@click.argument('test', nargs=1)
def eval(model, workers, device, valid_seq_len, seq_len, hidden, layers, kernel, batch_size, regularization, test):
    print('loading model')

    device = torch.device(device)
    model_state = torch.load(model)
    model = ConvSeqNet(model_state['oh_dim'],
                       model_state['oh_dim'],
                       model_state['hidden'],
                       model_state['layers'],
                       model_state['kernel'],
                       reg=model_state['regularization'])
    model.load_state_dict(model_state['state_dict'], map_location=lambda storage, loc: storage)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    print('loading test set')

    test_set = TextSet(glob.glob('{}/**/*.txt'.format(test), recursive=True), chars=defaultdict(lambda: 0, model_state['chars']))
    test_data_loader = DataLoader(dataset=test_set, num_workers=workers, batch_size=batch_size, pin_memory=True)

    loss = 0.0
    with torch.no_grad():
        with click.progressbar(test_data_loader, label='test') as bar:
            for sample in bar:
                input, target = sample[0].to(device), sample[1].to(device)
                o = model(input)
                o = o[:, :, :(seq_len - valid_seq_len)].contiguous()
                target = target[:, :(seq_len - valid_seq_len)].contiguous()
                loss += criterion(o, target)
    val_loss = loss / len(test_data_loader)
    print("===> bpc: {:.4f} (ppl char/word: {:.4f}/{:.4f})".format(1/np.log(2)*val_loss,
                                                                   np.exp(val_loss),
                                                                   np.exp(val_loss*test_set.avg_word_len())))


@cli.command()
@click.option('-n', '--name', default=None, help='prefix for checkpoint file names')
@click.option('-B', '--batch-size', default=1, show_default=True, help='minibatch size')
@click.option('-l', '--lrate', default=1e-3, show_default=True, help='initial learning rate')
@click.option('-e', '--weight-decay', show_default=True, default=0.0, help='Weight decay')
@click.option('-w', '--workers', default=0, show_default=True, help='number of workers loading training data')
@click.option('-d', '--device', default='cpu', show_default=True, help='pytorch device')
@click.option('-v', '--validation', default='val', show_default=True, help='validation set location')
@click.option('--lag', show_default=True, default=20, help='Number of epochs to wait before stopping training without improvement')
@click.option('--min-delta', show_default=True, default=0.005, help='Minimum improvement between epochs to reset early stopping')
@click.option('--optimizer', show_default=True, default='Adam', type=click.Choice(['SGD', 'Adam']), help='optimizer')
@click.option('--clip', show_default=True, default=0.0, help='gradient clipping value')
@click.option('--threads', show_default=True, default=1)
@click.option('-h', '--line-height', show_default=True, default=40, help='Height to normalize input lines to')
@click.argument('ground_truth', nargs=1)
def train(name, batch_size, lrate, weight_decay, workers, device, validation, lag, min_delta, optimizer,
          clip, line_height, threads, ground_truth):

    if not name:
        name = '{}_{}_{}_{}_{}'.format(optimizer.lower(), lrate, weight_decay, clip, line_height)
    print('model output name: {}'.format(name))

    torch.set_num_threads(threads)
    transforms = generate_input_transforms(1, line_height, 0, 1, 16, False, False)

    train_set = GroundTruthDataset(im_transforms=transforms, preload=False)
    for im in glob.glob('{}/**/*.png'.format(ground_truth), recursive=True):
        train_set.add(im)
    train_set.encode()
    o_dim = train_set.codec.max_label()+1

    train_loader = InfiniteDataLoader(train_set,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=workers,
                                      pin_memory=True,
                                      collate_fn=collate_sequences)

    val_set = GroundTruthDataset(im_transforms=transforms, preload=False)
    for im in glob.glob('{}/**/*.png'.format(validation), recursive=True):
        val_set.add(im)
    val_set.no_encode()
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            num_workers=workers,
                            pin_memory=True,
                            collate_fn=collate_sequences)


    device = torch.device(device)

    print('loading network')

    model = ConvSeqNet(line_height, o_dim).to(device)
    print(model)
    criterion = nn.CTCLoss(reduction='sum', zero_infinity=True)
    opti = getattr(torch.optim, optimizer)(model.parameters(), lr=lrate, weight_decay=weight_decay)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(opti, mode='max', patience=3)
    st_it = EpochStopping(50)
    epoch = 0
    while st_it.trigger():
        with click.progressbar(train_loader, label='epoch {}'.format(epoch), show_pos=True) as bar:
            for _, batch in zip(range(len(train_loader)), bar):
                input, target = batch['image'], batch['target']
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                input = input.requires_grad_()

                seq_lens, label_lens = batch['seq_lens'], batch['target_lens']
                seq_lens = seq_lens.to(device, non_blocking=True)
                label_lens = label_lens.to(device, non_blocking=True)
                output, seq_lens = model(input, seq_lens)

                opti.zero_grad()
                loss = criterion(output.permute(2, 0, 1),
                                 target,
                                 seq_lens,
                                 label_lens)

                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                del loss, output
            torch.save({'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'weight_decay': weight_decay,
                        'line_height': line_height,
                        'chars': dict(train_set.alphabet)}, '{}_{}.ckpt'.format(name, epoch))
            model.eval()
            seq_rec = TorchSeqRecognizer(model, train_set.codec, device=device)
            chars, error = compute_error(seq_rec, val_loader)
            model.train()
            lr_sched.step((chars-error)/chars)
            print("===> epoch {}: character accuracy: {})".format(epoch, (chars-error)/chars))
        epoch += 1

if __name__ == '__main__':
    cli()
