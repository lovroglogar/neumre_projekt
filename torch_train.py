import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter


def prepare_dataset(ds_train, ds_test, batch_size, train_val_split=0.8):
    torch.manual_seed(235)
    (ds_train, ds_val) = random_split(ds_train, [train_val_split, 1 - train_val_split], generator=torch.Generator())

    train_data_loader = DataLoader(ds_train, shuffle=True, batch_size=batch_size)
    val_data_loader = DataLoader(ds_val, shuffle=True, batch_size=batch_size)
    test_data_loader = DataLoader(ds_test, shuffle=True, batch_size=batch_size)

    return train_data_loader, val_data_loader, test_data_loader


def train_torch_model(model, train_data_loader, val_data_loader, config, device='cpu', write_to_tensorboard=True,
                      model_save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'lr': []
    }

    loss_function = nn.CrossEntropyLoss()

    train_steps = len(train_data_loader.dataset)
    val_steps = len(val_data_loader.dataset)

    if write_to_tensorboard:
        tb = SummaryWriter()

    for e in range(config['max_epochs']):
        model.train()

        train_loss = 0
        validation_loss = 0

        train_T = 0
        validation_T = 0

        for x, y in train_data_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_function(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_T += (pred.argmax(1) == y).type(torch.float).sum().item()

        with torch.no_grad():
            model.eval()
            for i, (x, y) in enumerate(val_data_loader):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                validation_loss += loss_function(pred, y)
                validation_T += (pred.argmax(1) == y).type(torch.float).sum().item()

        average_train_loss = train_loss / train_steps
        train_accuracy = train_T / train_steps

        average_validation_loss = validation_loss / val_steps
        validation_accuracy = validation_T / val_steps

        print(f'Epoch {e}, train loss={average_train_loss}, train accuracy={train_accuracy}')
        print(f'Epoch {e}, validation loss={average_validation_loss}, validation accuracy={validation_accuracy}')

        if write_to_tensorboard:
            tb.add_scalar('Training loss', average_train_loss, e)
            tb.add_scalar('Validation loss', average_validation_loss, e)
            tb.add_scalar('Training accuracy', train_accuracy, e)
            tb.add_scalar('Validation accuracy', validation_accuracy, e)

        history['train_loss'].append(average_train_loss.cpu().detach().numpy())
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(average_validation_loss.cpu().detach().numpy())
        history['val_accuracy'].append(validation_accuracy)

    if write_to_tensorboard:
        tb.close()

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)

    return history


def evaluate_model(model, data_loader, labels, device='cpu'):
    lf = nn.CrossEntropyLoss()
    num_classes = len(labels)
    with torch.no_grad():
        model.eval()
        M = np.zeros((num_classes, num_classes))
        loss = 0
        true = 0
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss += lf(pred, y)
            pred = pred.argmax(1)
            true += (pred == y).type(torch.float).sum().item()
            M += confusion_matrix(y.cpu().detach().numpy(), pred.cpu().detach().numpy(), labels=labels)

    print(f'Total loss: {loss / len(data_loader.dataset)}. Accuracy: {true / len(data_loader.dataset)}')
    print(M)
    pr = []
    for i in range(num_classes):
        tp_i = M[i, i]
        fn_i = np.sum(M[i, :]) - tp_i
        fp_i = np.sum(M[:, i]) - tp_i
        tn_i = np.sum(M) - fp_i - fn_i - tp_i
        recall_i = tp_i / (tp_i + fn_i)
        precision_i = tp_i / (tp_i + fp_i)
        print(f'Preciznost klase {i}: {precision_i}. Odziv klase {i}: {recall_i}')
        pr.append((precision_i, recall_i))

    return M.T, np.array(pr)
