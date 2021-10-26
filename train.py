from parabola_dataset import ParabolaDataset
from cnn import Net

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np


if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name()
    print(f"Running on your {gpu_name} (GPU)")
else:
    device = torch.device("cpu")
    print("Running on your CPU")

net = Net().to(device)


input = torch.randn(1, 1, 120, 160).to(device)
out = net(input)
net.zero_grad()
out.backward(torch.randn(1, 1, 120, 160).to(device))

parabola_ds = ParabolaDataset('parabolas', 'intercepts', 'filenames.csv')

train_size = int(0.8 * len(parabola_ds))
test_size = len(parabola_ds) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(parabola_ds, [train_size, test_size])

trainloader = DataLoader(train_dataset, batch_size=4,
                         shuffle=True, num_workers=2)

criterion = nn.L1Loss(reduction='sum')

# optimizer = optim.SGD(net.parameters(), lr=.001)
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

for epoch in range(2):  # loop over the dataset multiple times
    print('Beginning Epoch {}'.format(epoch))
    print('Learning rate: {}'.format(scheduler.get_last_lr()))
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['parabola'].to(device)
        labels = data['intercept'].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 90:    # print every 2000 mini-batches
            print('    batches: %5d loss: %.20f' %
                  (i + 1, running_loss / 2000))
            running_loss = 0.0
    scheduler.step()

print('Finished Training')
torch.save(net.state_dict(), 'trained_model.pt')

for data in train_dataset:
    out = np.where(net(data['parabola'].reshape((1, 1, 120, 160)).to(device)).detach().cpu().numpy().reshape((120, 160)) > 0.01, 1, 0)
    label = np.where(data['intercept'].numpy().reshape((120, 160)) > 0, 1, 0)
    parabola = data['parabola'].numpy().reshape((120, 160))

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.imshow(label, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Label')
    ax2.imshow(out, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Prediction')
    ax3.imshow(parabola, cmap='gray', vmin=0, vmax=1)
    ax3.set_title('Parabola')
    plt.show()
