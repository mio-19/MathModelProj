from __future__ import division
import time
import copy
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from torch import cuda

def learning_scheduler(optimizer, epoch, lr=0.001, lr_decay_epoch=10):
    """
    Learning scheduler for the optimizer
    Args:
        optimizer: optimizer object
        epoch: current epoch
        lr: initial learning rate
        lr_decay_epoch: epoch to decay learning rate
    Returns:
        optimizer: updated optimizer
    """
    lr = lr * (0.5**(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('Learning rate is set to {}'.format(lr))
    for param in optimizer.param_groups:
        param['lr'] = lr
    return optimizer

def train(model, train_loader, criterion, optimizer, init_lr=0.001, decay_epoch=10, n_epoch=20, use_cuda=True):
    """
    Train the model
    Args:
        model: model object
        train_loader: train data loader
        criterion: loss function
        optimizer: optimizer object
        init_lr: initial learning rate
        decay_epoch: epoch to decay learning rate
        n_epoch: number of epochs
        use_cuda: use cuda or not
    Returns:
        best_model: trained model
        loss_history: loss history
    """
    if use_cuda:
        model = model.cuda()
    best_model = model
    best_accuracy = 0.0
    loss_curve = []
    since = time.time()

    # Training
    for epoch in range(n_epoch):
        print('Epoch {}/{}'.format(epoch+1, n_epoch))
        
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = learning_scheduler(optimizer, epoch, lr=init_lr, lr_decay_epoch=decay_epoch)
                model.train(True)
            else:
                model.train(False)  
            
            # Reset the running loss and corrects
            # The running loss is used to calculate the average loss
            # The running corrects is used to calculate the average accuracy
            running_loss = 0.0
            running_corrects = 0 
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data
                _, predicted = torch.max(outputs.data, 1)
                running_corrects += torch.sum(predicted==targets.data).item()
                loss_curve.append(loss.data)

            # Calculate the average loss and accuracy
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_accuracy = running_corrects / len(train_loader.dataset)
            print('{} loss: {:.4f}, accuracy: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

            # Save the model if the accuracy is the best
            if phase == 'val' and epoch_accuracy>best_accuracy:
                best_accuracy = epoch_accuracy
                best_model = copy.deepcopy(model)
        print(' ')

    # Calculate the total time
    time_elapsed = time.time() - since
    print('{} trained in {:.0f}m {:.0f}s'.format(best_model.name, time_elapsed//60, time_elapsed%60))
    print('Best validation accuracy: {:.4f}'.format(best_accuracy))
    return best_model, loss_curve

def test(model, test_loader):
    """
    Test the model
    Args:
        model: model object
        test_loader: test data loader
    Returns:
        accuracy: accuracy
    """
    # corrects is the number of correct predictions
    corrects = 0
    model = model.cpu()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # The model is evaluated on the test set
        # The predictions are compared to the targets
        with torch.no_grad():
            # Convert the inputs and targets to Variable
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            # Calculate the accuracy
            _, predicted = torch.max(outputs.data, 1)
            corrects += torch.sum(predicted==targets.data).item()
    accuracy = corrects / len(test_loader.dataset)
    return accuracy
