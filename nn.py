#!python
import torch

import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


class Network(nn.Module):
    def __init__(self, input_num, hidden_size, out_put):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, out_put)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(hidden_size, out_put)

    def forward(self, x):
        out = self.tree_forward(x)
        return out

    def tree_forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


criterion = nn.CrossEntropyLoss()


class Node:
    def __init__(self, label: int, raw):
        self.label = label
        self.childs = []
        self.raw = raw

    def append(self, x):
        assert(x.label != self.label)
        self.childs.append(x)
        # self.child_data = torch.cat((self.child_data, x.raw), 0)


class Tree:
    def __init__(self):
        self.size = 0
        self.root = None

    def find_target(self, the_data: torch.Tensor, the_label: int, net: Network):
        path = []
        net.train(False)
        the_y = net(the_data)
        if self.root == None:
            return [], None

        iter = self.root
        y = net.tree_forward(iter.raw)
        min_value = torch.dist(the_y, y).item()
        min_index = None
        min_node = iter
        # print(min_value)
        while True:
            for index, child in enumerate(iter.childs):
                y = net(child.raw)
                value = torch.dist(the_y, y).item()
                if value < min_value:
                    min_value = value
                    min_node = child
                    min_index = index
                    # print(min_value)

            if min_node == iter:
                return (path, iter)
            path.append((iter, min_index))
            iter = min_node

    def insert(self, the_data: torch.Tensor, the_label: int, net: Network):
        # set net as test mode
        # assert(net.training == False)
        _, tail = self.find_target(the_data, the_label, net)
        if tail == None:
            self.root = Node(the_label, the_data)
            self.size += 1
            return

        if tail.label == the_label:
            return
        else:
            tail.append(Node(the_label, the_data))
            self.size += 1
            return

    def train(self, the_data, the_label, net: Network):
        path, tail = self.find_target(the_data, the_label, net)
        # final use multihot
        # construct tensor
        assert(tail != None)
        if len(path) == 0:
            # too rare, don't care
            return None, tail.label
        loss = 0

        the_y = net(the_data)
        net.train(True)
        for parent, index in path[:-1]:
            assert(index < len(parent.childs))
            logits = -torch.dist(net(parent.raw), the_y).view(1)
            for child in parent.childs:
                dis = -torch.dist(net(child.raw), the_y)
                logits = torch.cat((logits, dis.view(1)), 0)
            loss = loss + criterion(logits.view(1, -1),
                                    torch.LongTensor([index + 1]))

        loss = 0
        final_parent, _ = path[len(path) - 1]
        mask = [the_label == child.label for child in final_parent.childs]
        mask = [the_label == final_parent.label] + mask

        mask = torch.tensor(mask, dtype=torch.float)
        mask = Variable(mask)

        logits = -torch.dist(net(final_parent.raw), the_y).view(1)
        for child in final_parent.childs:
            dis = -torch.dist(net(child.raw), the_y)
            logits = torch.cat((logits, dis.view(1)), 0)
        prob = nn.Softmax(0)(logits)
        loss = loss + -torch.log(torch.max(torch.dot(prob, mask), torch.Tensor([1e-10])))

        return loss, tail.label


if __name__ == "__main__":
    input_size = 28 * 28
    num_classes = 10
    num_epochs = 10
    batch_size = 1
    learning_rate = 1e-3

    train_dataset = dsets.MNIST(
        root='./mnist', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(
        root='./mnist', train=False, transform=transforms.ToTensor(), download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    hidden_size = 400

    model = Network(input_size, hidden_size, 20)

    learning_rate = 1e-4
    num_epoches = 5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # for i, (images, labels) in enumerate(train_loader):
    #     if i > 1000:
    #         break
    #     images = Variable(images.view(-1, 28 * 28))
    #     # print(images.shape)
    #     labels = Variable(labels)

    #     optimizer.zero_grad()
    #     outputs = model.forward(images)
    #     loss = criterion(outputs, labels)
    #     loss.backward()
    #     optimizer.step()
    #     if i % 100 == 0:
    #         print('current loss = %.5f' % loss.data)
    limit = 1000
    tree_size = 1
    step = 1
    while True:
        tree = Tree()
        # construct tree
        for i, (image, label) in enumerate(train_loader):
            image = Variable(image.view(1, 28*28))
            label = label.item()
            tree.insert(image, label, model)
            # print(tree.size, i)
            tree_size = tree.size
            if i >= 1000:
                break

        # # optimize tree
        total = 0
        acc = 0
        acc_sum = 0.0
        acc_cnt = 0
        for i, (image, label) in enumerate(train_loader):
            if i > limit:
                break
            optimizer.zero_grad()
            image = Variable(image.view(1, 28*28))
            label = label.item()
            loss, the_label = tree.train(image, label, model)
            total += 1
            acc += label == the_label
            if loss and loss.data < 10:
                loss.backward(retain_graph=True)
                optimizer.step()
                vloss = loss.data
                if i > 980:
                    acc_sum += (100.0 * acc / total)
                    acc_cnt += 1
                # print(i, ' cur loss = %.5f ' % vloss, 'avg acc = %.5f%%' % (100.0 * acc / total))
            # else:
            #     print(i, ' cur loss = SKIP ', 'avg acc = %.5f%%' % (100.0 * acc / total))
        # limit += 1000
        print(f'step:{step}, tree_size:{tree_size}, acc:{acc_sum / acc_cnt if acc_cnt != 0 else 100.0 * acc / total}')
        step += 1
    total = 0
    correct = 0

    # for images, labels in test_loader:
    #     images = Variable(images.view(-1, 28 * 28))
    #     outputs = model(images)

    #     _, predicts = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicts == labels).sum().item()

    print(total)
    print(correct)
    # print('Accuracy = %.2f' % (100.0 * correct / total))
