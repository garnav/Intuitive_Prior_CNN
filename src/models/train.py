# train.py
# Arnav Ghosh
# 3rd Aug. 2019

import time
import torch

######### CONSTANTS #########
BATCH_SIZE = 32
EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Params(object):
    pass

# class Params(object):
#     def __init__(self, batch_size, test_batch_size, epochs, lr, momentum, seed, cuda, log_interval):
#         self.batch_size = batch_size
#         self.test_batch_size = test_batch_size
#         self.epochs = epochs
#         self.lr = lr
#         self.momentum = momentum
#         self.log_interval = log_interval

def get_model():
    pass

def train_net(params, dataloaders, model, criterion, optimizer):
    begin = time.time()

    best_weights = best_wgts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(params.num_epochs):
        print(f'Epoch {epoch + 1}/{params.num_epochs}')
        print('=' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            loss = 0.0
            acc = 0.0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    l = criterion(outputs, labels)

                    if phase == 'train':
                        l.backward()
                        optimizer.step()

                loss += l.item() * inputs.size(0) #de-average to mitigate batch_num bias
                accuracy += torch.sum(preds == labels.data)

            epoch_loss = loss / len(dataloaders[phase])
            epoch_acc = accuracy.double() / len(dataloaders[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - begin
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    # params
    # model
    # loss
    # optimizer
    # train
    pass