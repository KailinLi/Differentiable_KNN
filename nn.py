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
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, out_put)
        

    def forward(self, x):
        out = self.tree_forward(x)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

    def tree_forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

criterion = nn.CrossEntropyLoss()

class Node:
    def __init__(self, label:int, out):
        self.label = label 
        self.childs = []    
        self.out = out
        # self.child_data = torch.Tensor(0, 100)

    def append(self, x):
        assert(x.label != self.label)
        self.childs.append(x)
        # self.child_data = torch.cat((self.child_data, x.out), 0)
        
class Tree: 
    def __init__(self):
        self.size = 0
        self.root = None

    def insert(self, the_data:torch.Tensor, the_label:int, net:Network):
        # set net as test mode
        # assert(net.training == False)

        net.train(False)
        the_y = net.tree_forward(the_data) 
        if self.root == None:
            self.root = Node(the_label, the_y)
            self.size += 1
            return 

        iter = self.root
        min_value = torch.dist(the_y, iter.out).item()
        # print(min_value)
        while True: 
            min_node = iter
            for child in iter.childs:
                value = torch.dist(the_y, child.out).item()
                if value < min_value:
                    min_value = value
                    min_node = child
                    # print(min_value)

            if min_node == iter:
                if iter.label == the_label:
                    # hit
                    return
                else:
                    iter.append(Node(the_label, the_y))
                    self.size += 1
                    return
            iter = min_node 

    def train(self, the_data, the_label, net:Network):
        path = []
        indexes = []
        final_iter = None
        net.train(True)
        the_y = net.tree_forward(the_data)
        iter = self.root
        min_value = torch.dist(the_y, iter.out).item()
        # print(min_value)
        while True: 
            min_node = iter
            min_i = -1
            for i in range(len(iter.childs)):
                child = iter.childs[i]
                value = torch.dist(the_y, child.out).item()
                if value < min_value:
                    min_value = value
                    min_node = child
                    # print(min_value)
            if min_node == iter:
                final_iter = iter
                break 
            path.append(iter)
            indexes.append(i)
            iter = min_node 

        # final use multihot
        # construct tensor 
        assert(final_iter != None)
        loss = 0
        mask = [the_label == child.label for child in final_iter.childs]
        if len(mask) == 0:
            pass
        else:
            mask = torch.tensor(mask, dtype=torch.float) 
            mask = Variable(mask)
            # print(mask)
            logits = None
            for child in final_iter.childs:
                dis = -torch.dist(child.out, the_y)
                # print(dis)
                if logits is None:
                    logits = dis.view(1)
                else:
                    logits = torch.cat((logits, dis.view(1)), 0)
            prob = nn.Softmax(0)(logits)
            # print(prob)
            loss = torch.log(torch.dot(prob, mask))

        for i in range(len(path)):
            iter = path[i]
            index = indexes[i]
            assert(index < len(iter.childs))
            logits = None
            for child in iter.childs:
                dis = -torch.dist(child.out, the_y)
                print(dis)
                if logits is None:
                    logits = dis.view(1)
                else:
                    logits = torch.cat((logits, dis.view(1)), 0)
            loss = loss + criterion(logits.view(1, -1), torch.LongTensor([index]))
        return loss  
        
    
        

if __name__ == "__main__":
    input_size = 28 * 28
    num_classes = 10
    num_epochs = 10
    batch_size = 1
    learning_rate = 1e-3

    train_dataset = dsets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor(), download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=True)



    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    hidden_size = 100

    model = Network(input_size, hidden_size, num_classes)

    learning_rate = 1e-5
    num_epoches = 5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for i, (images, labels) in enumerate(train_loader):
        if i > 1000:
            break
        images = Variable(images.view(-1, 28 * 28))
        # print(images.shape)
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('current loss = %.5f' % loss.data)

    while True:
        tree = Tree()
        # construct tree
        for i, (image, label) in enumerate(train_loader):
            image = Variable(image.view(1, 28*28))
            label = label.item() 
            tree.insert(image, label, model)
            if tree.size > 1000:
                break
    
        # optimize tree 
        for i, (image, label) in enumerate(train_loader):
            if i > 10000:
                break

            optimizer.zero_grad()
            image = Variable(image.view(1, 28*28))
            label = label.item() 
            loss = tree.train(image, label, model)
            loss.backward(retain_graph=True) 
            optimizer.step()
        
            vloss = loss.data
            print('cur loss = %.5f' % vloss)


    total = 0
    correct = 0

    for images, labels in test_loader:
        images = Variable(images.view(-1, 28 * 28))
        outputs = model(images)

        _, predicts = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicts == labels).sum().item()

    print(total)
    print(correct)
    print('Accuracy = %.2f' % ( 100.0 * correct / total))