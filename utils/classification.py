import torch
from utils.utils import reset_net, get_neuron_threshold


# Testing pre-trained ANN -- CIFAR
def testing(model, testLoader, criterion, device):
    model.eval()  # Put the model in test mode

    running_loss = 0.0
    correct = 0
    total = 0
    for data in testLoader:
        inputs, labels = data

        # Transfer to GPU
        inputs, labels = inputs.type(torch.FloatTensor).to(device), \
                         labels.type(torch.LongTensor).to(device)

        # forward pass
        _, y_pred = model.forward(inputs)
        loss = criterion(y_pred, labels)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    # calculate epoch statistics
    epoch_loss = running_loss / len(testLoader)
    acc = correct / total

    return acc, epoch_loss

# Testing LTL-fine-tuned SNN -- CIFAR
def testing_snn_Burst(snn, testLoader, device, T):
    tot = torch.zeros(T).to(device)
    spk = [0] * T
    spk_cnt = [0] * T
    length = 0
    model = snn.to(device)
    model.eval()
    # evaluate
    threshold = get_neuron_threshold(snn)
    with torch.no_grad():
        for idx, (inputs, label) in enumerate(testLoader):
            spikes = 0
            length += len(label)
            inputs = inputs.to(device)
            label = label.to(device)
            for t in range(T):
                hidden, out = model(inputs, SNN=True)
                spikes += out
                tot[t] += (label==spikes.max(1)[1]).sum()
                spk[t] += (((hidden[-1] > 0).float()).sum()).item()
                spk_cnt[t] += (((hidden[-1]).float()).sum()).item()
            reset_net(model)

    spk_cnt = [(x / threshold[-1]) / length for x in spk_cnt]
    spk_cnt = ['%.2f' % n for n in spk_cnt]
    spk = [x/length for x in spk]
    return tot/length, spk,  spk_cnt

# TTFS learning testing -- CIFAR
def testing_snn_TTFS(snn, ttfs_model, testloader, device, sim_len, threshold):
    correct = torch.zeros(sim_len).to(device)
    total = 0
    ttfs_model.eval()
    snn = snn.to(device)
    snn.eval()
    avg_spk_time = torch.zeros(sim_len).to(device)
    hidden_threshold = get_neuron_threshold(snn)
    out_count = torch.zeros(sim_len).to(device)

    for i_batch, (inputs, labels) in enumerate(testloader, 1):
        # Transfer to GPU
        inputs = inputs.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.LongTensor).to(device)

        total += len(labels)
        spk_counts = []
        end_symbol = torch.zeros(inputs.size(0),1,1,1).to(device)

        for t in range(sim_len):
            inputs = inputs * (1 - end_symbol) # Once the first output spike is detected for a sample, the simulation of this sample ends with inputting zero.

            # Get the last hidden layers output
            _, out = snn.forward(inputs, SNN=True, TTFS=True) # spk and mem size [N, Class]
            spk_count = out / hidden_threshold[-1] # burst count of the last hidden layer
            spk_counts.append(spk_count)
            count_t = torch.stack(spk_counts).permute(1, 2, 0)

            # TTFS
            V_out, spikes = ttfs_model(threshold, count_t) # [N, T, Class]

            # Get t_f
            t_idx = torch.arange(0, t + 1).unsqueeze(0).unsqueeze(2).repeat(spikes.size(0), 1, spikes.size(2)).to(device)  # [N,t+1,10]
            timing = (t + 1 - t_idx) * spikes
            boolFire = spikes.sum(dim=1, keepdim=True) > 0
            t_f = torch.argmax(timing.float(), dim=1, keepdim=True)  # [N,1,Class]
            t_last = torch.ones_like(t_f) * t
            t_f = torch.where(boolFire, t_f, t_last)  # Once one neuron never fire, it will be considered as fire at the last timestep t
            t_f = torch.min(t_f, 2)[0].unsqueeze(2).repeat(1, 1, t_f.size(2))

            V_t_f = torch.gather(V_out, 1, t_f).squeeze(1)
            correct[t] += (labels == V_t_f.argmax(dim=1)).sum().item()
            avg_spk_time[t] += t_f.float().sum() / t_f.size(2)
            end_symbol = (boolFire.sum(dim=2) > 0).view(-1,1,1,1).float() # The samples that detect the first end.

        reset_net(snn)

        # Distribution Analysis
        for t in range(sim_len):
            out_count[t] += (t_f[:,-1,-1] == t).float().sum().item()

    return correct.max()/total, (avg_spk_time[torch.argmax(correct)]/total) + 1, out_count/total



# Training pre-trained ANN -- CIFAR
def training(model, trainloader, optimizer, criterion, device):
    model.train()  # Put the model in train mode

    running_loss = 0.0
    total = 0
    correct = 0
    #cnt =0
    for i_batch, (inputs, labels) in enumerate(trainloader, 1):

        # Transfer to GPU
        inputs, labels = inputs.type(torch.FloatTensor).to(device), \
                         labels.type(torch.LongTensor).to(device)

        # Model computation and weight update
        _, y_pred = model.forward(inputs)
        loss = criterion(y_pred, labels)
        _, predicted = torch.max(y_pred.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / i_batch
    acc_train = correct / total

    return model, acc_train, epoch_loss


# LTL-fine-tuning SNN -- CIFAR
def training_thNorm_with_T(ann, snn, trainloader, optimizer, criterion_out, criterion_local, coeff_local, device, T, gamma=5):
    """SNN fine-tune with threshold normalization"""
    snn.train()  # Put the model in train mode
    ann.eval()

    running_loss = 0.0
    total = 0
    correct = 0

    for i_batch, (inputs, labels) in enumerate(trainloader, 1):
        # Transfer to GPU
        inputs, labels = inputs.type(torch.FloatTensor).to(device), \
                         labels.type(torch.LongTensor).to(device)

        # Model computation and weight update
        hiddenA, _ = ann.forward(inputs)
        y_pred = 0
        hiddenS = 0
        for t in range(T):
            out_ = snn.forward(inputs, SNN=True)
            hidden_cached = out_[0]
            y_pred += out_[1]
            if t == 0:# for the first step
                hiddenS = [x.clone() for x in hidden_cached]
            else:
                for iLayer in range(len(hidden_cached)):
                    hiddenS[iLayer] += hidden_cached[iLayer]
        hiddenS = [x/T for x in hiddenS]
        y_pred = y_pred / T

        loss = criterion_out(y_pred, labels)
        # Compute local loss
        for (A, S, C) in zip(hiddenA, hiddenS, coeff_local):
            loss += C * criterion_local(S, A*gamma)

        _, predicted = torch.max(y_pred.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        reset_net(snn)

    epoch_loss = running_loss / i_batch
    acc_train = correct / total

    return snn, acc_train, epoch_loss


# TTFS learning training -- CIFAR
def training_snn_TTFS(snn, ttfs_model, trainloader, optimizer, loss_fn, alpha, beta, device, sim_len, threshold):
    snn.eval()  # Put the model in test mode
    avg_spk_time = 0
    avg_spk_pro = 0
    running_loss = 0.0
    running_loss1 = 0.0
    running_loss2 = 0.0
    total = 0
    correct = 0
    hidden_threshold = get_neuron_threshold(snn)
    ttfs_model.train()

    for i_batch, (inputs, labels) in enumerate(trainloader, 1):
        # Transfer to GPU
        inputs = inputs.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.LongTensor).to(device)

        label_ = torch.zeros(len(inputs), ttfs_model.LIF.fc.out_features).to(device)
        label_ = label_.scatter_(1, labels.view(-1, 1), 1)
        spk_counts = []

        # Get the last hidden layers output
        for t in range(sim_len):
            _, out = snn.forward(inputs, SNN=True, TTFS=True)
            spk_count = out / hidden_threshold[-1]
            spk_counts.append(spk_count)

        count_t = torch.stack(spk_counts, dim=0).permute(1, 2, 0).detach()  # [N, hid_dim, T])
        reset_net(snn)

        # TTFS
        V_out, spikes = ttfs_model(threshold, count_t) # [N, T, Class]

        # Get t_f
        with torch.no_grad():
            t_idx = torch.arange(0, sim_len).unsqueeze(0).unsqueeze(2).repeat(spikes.size(0), 1, spikes.size(2)).to(device)
            timing = (sim_len - t_idx) * spikes
            boolFire = spikes.sum(dim=1, keepdim=True) > 0
            t_f = torch.argmax(timing, dim=1, keepdim=True)  # [N,1,Class]
            last = torch.ones_like(t_f) * (sim_len -1)
            t_f = torch.where(boolFire, t_f, last)  # Once one neuron never fire, it will be considered as fire at the last timestep
            t_f = torch.min(t_f, 2)[0].unsqueeze(2).repeat(1, 1, t_f.size(2))  # [N,1,Class]

        V_t_f = torch.gather(V_out, 1, t_f).squeeze(1)  # [N,10]

        # Computing loss
        LTD_mask = ((V_t_f >= threshold) * (1 - label_)).detach()
        theta = 1 - ((V_t_f.argmax(dim=1) == labels).sum() / (V_out[:,-1,:].argmax(dim=1) == labels).sum()).clamp(0,1).item() # Scaling factor

        loss1 = loss_fn(V_t_f, labels)
        loss2 = (theta * (V_t_f - threshold) * LTD_mask).sum()

        loss = alpha * loss1 + beta * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
        avg_spk_pro += boolFire.float().mean()
        avg_spk_time += t_f.float().sum() / t_f.size(2)
        correct += (V_t_f.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        if i_batch % 100 == 0:
            print('Batch {}, Train Acc {:.4f}, Train Loss {:.4f}, Avg spike time {:.4f}, spike probability {:.4f}'
                  .format(i_batch, (correct / total) * 100, running_loss / i_batch, (avg_spk_time / total) + 1,
                          avg_spk_pro / i_batch))

    epoch_loss = running_loss / i_batch
    epoch_loss1 = running_loss1 / i_batch
    epoch_loss2 = running_loss2 / i_batch
    print('Train loss: ', epoch_loss1, epoch_loss2)

    return ttfs_model, correct / total, epoch_loss

