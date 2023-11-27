import time
import os
import argparse
import sys
current_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(current_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CUDA configuration
import copy
import torch
import torch.nn as nn

from models.vgg16 import VGG16
from models.TTFS_LIF import TTFS_LIF_linear
from data.data_loader_cifar10 import build_data
from utils.classification import training_thNorm_with_T,testing_snn_Burst, testing, training_snn_TTFS,testing_snn_TTFS
from utils.utils import search_fold_and_remove_bn, replace_activation_by_neuron, replace_IF_by_Burst_all, get_maximum_activation
from utils.lib import dump_json, set_seed

set_seed(1111)

# Load datasets
home_dir = current_dir # relative path
data_dir = 'C:/dataset/raw' # Data dir
ann_ckp_dir = os.path.join(home_dir, 'exp/cifar10/')
snn_ckp_dir = os.path.join(home_dir, 'exp/cifar10/snn/')

parser = argparse.ArgumentParser(description='PyTorch Cifar-10 Training')
parser.add_argument('--Tencode', default=16, type=int, metavar='N',
                    help='encoding time window size')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=96, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr1', default=1e-4, type=float,
                    metavar='LR_S1', help='initial learning rate of LTL training', dest='lr1')
parser.add_argument('--lr2', default=1e-5, type=float,
                    metavar='LR_S2', help='initial learning rate of TTFS training', dest='lr2')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save the training record (default: none)')
parser.add_argument('--local_coefficient', default=1.0, type=float,
                     help='Coefficient of Local Loss')
parser.add_argument('--beta', default=2, type=float,
                    metavar='beta', help='coefficient beta')
parser.add_argument('--gamma', default=5, type=float,
                    metavar='gamma', help='Maximum number of spikes per timestep in burst coding')
parser.add_argument('--threshold', default=3, type=float,
                     help='The potential threshold of the output layer (TTFS coding)')
parser.add_argument('--ltl_resume', default=True, action='store_true',
					help='Resume from LTL finetuned model and start ttfs learning')


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
        print('GPU is available')
    else:
        device = 'cpu'
        print('GPU is not available')

    # Parameters
    args = parser.parse_args()
    Tencode = args.Tencode
    num_epochs = args.epochs
    lr_ltl = args.lr1
    lr_ttfs = args.lr2

    alpha = 2  # coefficient alpha
    beta = args.beta  # coefficient beta

    best_test_acc = 0
    best_test_spktime = Tencode
    batch_size = args.batch_size
    coeff_local = [args.local_coefficient] * 15 # Local loss coefficient for each layer
    test_acc_history = []
    train_acc_history = []
    test_spktime_history = []

    # Prepare dataloader
    train_loader, test_loader = build_data(dpath=data_dir, batch_size=batch_size, cutout=True, workers=0, use_cifar10=True, auto_aug=True)

    # Init ANN and load pre-trained model
    model = VGG16(num_class=10)
    model = model.to(device)
    TTFS_model = TTFS_LIF_linear(4096, 10).to(device)
    checkpoint = torch.load(ann_ckp_dir + 'checkpoint/vgg16_relu_wAvgPool_baseline.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Accuracy of pre-trained model {}'.format(checkpoint['acc']))
    search_fold_and_remove_bn(model)
    ann = copy.deepcopy(model)

    get_maximum_activation(train_loader, model=model, momentum=0.9, iters=20, mse=True, percentile=None,
                           sim_length=Tencode, channel_wise=False)

    # Init SNN model with ANN weights
    model = replace_activation_by_neuron(model)
    model = replace_IF_by_Burst_all(model, gamma=args.gamma)
    snn = copy.deepcopy(model)
    print(snn)

    # Training configuration
    criterion_out = torch.nn.CrossEntropyLoss()
    criterion_local = nn.MSELoss()  # Local loss function
    optimizer = torch.optim.Adam(snn.parameters(), lr=lr_ltl, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40, 50, 80, 90], gamma=0.5)

    # Testing ANN and SNN model
    acc_test, loss_test = testing(ann, test_loader, criterion_out, device)
    print('Accuracy of BN folded ANN model {}'.format(acc_test))

    acc_test, spk, spk_cnt = testing_snn_Burst(snn, test_loader, device, Tencode)
    print('Accuracy of converted SNN model {}'.format(acc_test))

    if not args.ltl_resume:
        """
            Stage1: Hidden Layers Training -- LTL
            """
        print('------ Stage 1 - Training Hidden Layers ------')

        for epoch in range(num_epochs):
            since = time.time()

            # Training Stage
            snn, acc_train, loss_train = training_thNorm_with_T(ann, snn, train_loader, optimizer, criterion_out,
                                                                criterion_local, coeff_local, device, Tencode, args.gamma)
            scheduler.step()

            # Testing Stage
            acc_test, spk, spk_cnt = testing_snn_Burst(snn, test_loader, device, Tencode)

            # log results
            test_acc_history.append(acc_test[-1].item())
            train_acc_history.append(acc_train)

            # Report Training Progress
            time_elapsed = time.time() - since
            print('Stage1, Epoch {:d} takes {:.0f}m {:.0f}s'.format(epoch + 1, time_elapsed // 60, time_elapsed % 60))
            print('Train Accuracy: {:4f}, Loss: {:4f}'.format(acc_train, loss_train))
            print('Test Accuracy: {}'.format(acc_test))

            # Save Model
            if acc_test[-1] > best_test_acc:
                print("Saving the model.")

                if not os.path.isdir(snn_ckp_dir + 'checkpoint'):
                    os.makedirs(snn_ckp_dir + 'checkpoint')

                state = {
                    'epoch': epoch,
                    'model_state_dict': snn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train,
                    'acc': acc_test[-1],
                }
                torch.save(state, snn_ckp_dir + 'DBD_CIFAR10_vgg16.pth')
                best_test_acc = acc_test[-1].item()

            print('Best Test Accuracy: {:4f}'.format(best_test_acc))

            training_record = {
                'test_acc_history': test_acc_history,
                'train_acc_history': train_acc_history,
                'best_acc': best_test_acc,
            }
            dump_json(training_record, snn_ckp_dir + 'record', 'cifar10_vgg16_record_LTL.pth')
    else:
        LTL = torch.load(snn_ckp_dir + 'DBD_CIFAR10_vgg16.pth')
        snn.load_state_dict(LTL['model_state_dict'])
        print('Resume the LTL-finetuned Model')
        acc_test, spk, spk_cnt = testing_snn_Burst(snn, test_loader, device, Tencode)
        print('Stage1 Test Accuracy: {}'.format(acc_test))

    """
    Stage2: Output Layers Training -- TTFS
    """
    print('------ Stage 2 - Training Output Layer ------')

    optimizer = torch.optim.SGD(TTFS_model.parameters(), lr=lr_ttfs, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.3 * num_epochs, 0.5 * num_epochs, 0.8 * num_epochs],gamma=0.5)

    for epoch in range(num_epochs):

        # Training Stage
        since = time.time()
        TTFS_model, acc_train, loss_train = training_snn_TTFS(snn, TTFS_model, train_loader, optimizer, criterion_out, alpha, beta, device, Tencode, args.threshold)
        scheduler.step()

        # Testing Stage
        acc_test, avg_test_time, spk_count = testing_snn_TTFS(snn, TTFS_model, test_loader, device, Tencode, args.threshold)

        # Report Training Progress
        time_elapsed = time.time() - since
        print('Stage2, Epoch {:d} takes {:.0f}m {:.0f}s'.format(epoch + 1, time_elapsed // 60, time_elapsed % 60))
        print('Train Accuracy: {:.4f}, Loss: {:.4f}'.format(acc_train, loss_train))
        print('Test Accuracy: {:.4f}, Test spiking time: {:.4f}'.format(acc_test.item(),avg_test_time.item()))
        #print('Test spiking distribution: {}'.format(spk_count)) # print the spike distribution

        # log results
        test_acc_history.append(acc_test.item())
        test_spktime_history.append(avg_test_time.item())
        train_acc_history.append(acc_train)


        # Save Model
        if acc_test > best_test_acc:
            print("Saving the model.")
            torch.save(TTFS_model.state_dict(), snn_ckp_dir + 'DBT_CIFAR10_vgg16.pth')
            if not os.path.isdir(snn_ckp_dir + 'checkpoint'):
                os.makedirs(snn_ckp_dir + 'checkpoint')

            state = {
                'epoch': epoch,
                'model_state_dict': TTFS_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_train,
                'acc': acc_test,
            }
            best_test_acc = acc_test
            best_test_spktime = avg_test_time
        print('Accuracy: {:.4f}, Average spike time: {:.4f}, Best acc: {:.4f}, Best acc spike time: {:.4f}'.format(acc_test,avg_test_time,best_test_acc, best_test_spktime))

        training_record = {
            'test_acc_history': test_acc_history,
            'test_spiketime_history': test_spktime_history,
            'train_acc_history': train_acc_history,
            'best_acc': best_test_acc.item(),
        }
        dump_json(training_record, snn_ckp_dir + 'record', 'cifar10_vgg16_record_TTFS.pth')