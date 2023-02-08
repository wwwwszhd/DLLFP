import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, criterion, optimizer, train_loader, epoch, num_epoch):
    model.train()
    batch_loss = []
    batch_acc = []
    for step, (train_x, train_y) in enumerate(train_loader):
        train_x = train_x.to(device)
        train_y = torch.eye(2)[train_y].to(device)
        output = model(train_x)
        loss = criterion(output, train_y)
        batch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        acc = (output.argmax(1) == train_y.argmax(1)).float().mean()
        batch_acc.append(acc)
        # print('Train : Epoch [{}/{}], Loss: {:.4f}, acc: {:.2f}' .format(epoch + 1, num_epoch, loss.item(), acc.item()))

    batch_loss = sum(batch_loss) / len(batch_loss)
    batch_acc = sum(batch_acc) / len(batch_acc)
    print('Train : Epoch [{}/{}], Train Loss: {:.4f}, Train acc: {:.2f}' .format(epoch + 1, num_epoch, batch_loss, batch_acc))
    return batch_loss

def eval(model, criterion, optimizer, val_loader, epoch, num_epoch):
    model.eval()
    batch_loss = []
    batch_acc = []
    with torch.no_grad():
        for step, (val_x, val_y) in enumerate(val_loader):
            val_x = val_x.to(device)
            val_y = torch.eye(2)[val_y].to(device)
            optimizer.zero_grad()
            output = model.forward(val_x)
            loss = criterion(output, val_y)
            batch_loss.append(loss.item())

            acc = (output.argmax(1) == val_y.argmax(1)).float().mean()
            batch_acc.append(acc)
            # print('Valid : Epoch [{}/{}], Loss: {:.4f}, acc: {:.2f}' .format(epoch + 1, num_epoch, loss.item(), acc.item()))

    batch_loss = sum(batch_loss) / len(batch_loss)
    batch_acc = sum(batch_acc) / len(batch_acc)
    print('Valid : Epoch [{}/{}], Valid Loss: {:.4f}, Valid acc: {:.2f}' .format(epoch + 1, num_epoch, batch_loss, batch_acc))
    return batch_loss

def test(model, optimizer, test_loader):
    model.eval()
    with torch.no_grad():
        for step, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_y = F.one_hot(test_y, num_classes=2)
            test_y = test_y.to(device)
            optimizer.zero_grad()
            output = model.forward(test_x)
            pred = output.argmax(1).cpu().numpy()
            true = test_y.argmax(1).cpu().numpy()
            proba = output[:, 1].cpu().numpy()

        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred)
        sensitivity = recall_score(true, pred)
        f1score = f1_score(true, pred)
        MCC = matthews_corrcoef(true, pred)
        AUC = roc_auc_score(true, proba)
        print('ACC: {:.4f}, PRE: {:.4f}, SEN: {:.4f}, F1: {:.4f}, MCC: {:.4f}, AUC: {:.4f}' .format(acc, precision, sensitivity, f1score, MCC, AUC))
        fpr, tpr, thresholds = roc_curve(true, proba)
        print(thresholds)
        plt.plot(fpr, tpr, label='proposed method' + ' (AUC=%6.3f) ' % AUC)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()