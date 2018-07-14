import torch
import sys
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def prune_net(l1,l2):
    o_layer = copy.deepcopy(l1)
    o_weights2 = copy.deepcopy(l2).weight.data

    eliminate = []
    for i in range(0,len(o_layer.weight.data)):
        if sum(o_layer.weight.data[i]) < -1:
            eliminate.append(i)

    l1 = nn.Linear(o_layer.in_features, o_layer.out_features-len(eliminate))
    l2 = nn.Linear(l1.out_features, l2.out_features)
    last_append = 0
    for n in range(0,len(o_layer.weight.data)):
        if n not in eliminate:
            l1.weight.data[last_append] = o_layer.weight.data[n]
            last_append += 1

    for n in range(0,len(o_weights2)):
        last_append = 0
        for y in range(0,len(o_weights2[n])):
            if y not in eliminate:
                l2.weight.data[n][last_append] = o_weights2[n][y]
                last_append += 1
    return l1, l2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


net = Net()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(6):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

            if sys.argv[1] == 'prune':
                net.fc1, net.fc2 = prune_net(net.fc1,net.fc2)
                net.fc2, net.fc3 = prune_net(net.fc2,net.fc3)
                net.fc3, net.fc4 = prune_net(net.fc3,net.fc4)
                optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
                criterion = nn.CrossEntropyLoss()
                print(net)

print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()


outputs = net(images)

_, predicted = torch.max(outputs, 1)


correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
            correct += c[i].item()
            total += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))








#
