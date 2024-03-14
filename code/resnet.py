import pytorch_lightning as pl
import preprocess
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import Trainer
from torchvision import models
from torch import nn
from torchmetrics import Precision, Recall, Accuracy
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import  interp


class LitResNet(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-4):
        super().__init__()
        self.model = models.resnet101(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        # Metrics
        self.accuracy = Accuracy(num_classes=num_classes, average='none', task='multiclass')
        self.precision = Precision(num_classes=num_classes, average='none', task='multiclass')
        self.recall = Recall(num_classes=num_classes, average='none', task='multiclass')
        self.accuracy_sum = Accuracy(num_classes=num_classes, average='none', task='multiclass')
        # plotting
        self.predictions = []
        self.true_labels = []
        self.train_losses = []
        self.val_losses = []
        self.class_accuracies = {i:[] for i in range(self.num_classes)}
        self.overall_accuracies_train = []
        self.overall_accuracies_val = []
        
        
    def forward(self, x):
        x = self.model(x)
        x = F.softmax(x, dim=1)
        return x
        
        
    def step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        return y_hat, y, loss


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y).mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y).mean()
        acc_class0 = self.accuracy(y_hat, y)[0]
        acc_class1 = self.accuracy(y_hat, y)[1]
        acc_class2 = self.accuracy(y_hat, y)[2]
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_class0', acc_class0, on_step=False, on_epoch=True,  logger=True)
        self.log('val_acc_class1', acc_class1, on_step=False, on_epoch=True,  logger=True)  
        self.log('val_acc_class2', acc_class2, on_step=False, on_epoch=True,  logger=True)
        return loss


    def test_step(self, batch, batch_idx):
        y_hat, y, loss = self.step(batch)
        # Log metrics
        accs = self.accuracy(y_hat, y)
        precs = self.precision(y_hat, y)
        recs = self.recall(y_hat, y)
        acc_sum = self.accuracy_sum(y_hat, y).mean()

        # Calculate overall precision and recall
        overall_prec = self.precision(y_hat, y).mean()
        overall_rec = self.recall(y_hat, y).mean()

        self.predictions.append(y_hat.detach().cpu().numpy())
        self.true_labels.append(y.detach().cpu().numpy())
        
        for i, (acc, prec, rec) in enumerate(zip(accs, precs, recs)):
            # Determine the label based on the index i
            if i == 0:
                label_suffix = '9-10'
            elif i == 1:
                label_suffix = '10-11'
            else:
                label_suffix = '11-12'
            
            # Log metrics with the determined label suffix
            self.log(f'test_acc_{label_suffix}', acc, prog_bar=True, logger=True)
            self.log(f'test_prec_{label_suffix}', prec, prog_bar=True, logger=True)
            self.log(f'test_rec_{label_suffix}', rec, prog_bar=True, logger=True)

        # Log overall precision and recall
        self.log('test_overall_prec', overall_prec, prog_bar=True, logger=True)
        self.log('test_overall_rec', overall_rec, prog_bar=True, logger=True)
        
        self.log('test_acc_sum', acc_sum, prog_bar=True, logger=True)
        self.log('test_loss', loss, prog_bar=True, logger=True)



    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def on_train_epoch_end(self):
        # Retrieve the average training loss for the epoch
        train_loss_avg = self.trainer.callback_metrics.get('train_loss')
        if train_loss_avg is not None:
            self.train_losses.append(train_loss_avg.item())
        self.overall_accuracies_train.append(self.trainer.callback_metrics.get('train_acc').item())


    def on_validation_epoch_end(self):
        # Retrieve the average validation loss for the epoch
        val_loss_avg = self.trainer.callback_metrics.get('val_loss')
        if val_loss_avg is not None:
            self.val_losses.append(val_loss_avg.item())
        self.class_accuracies[0].append(self.trainer.callback_metrics.get('val_acc_class0').item())
        self.class_accuracies[1].append(self.trainer.callback_metrics.get('val_acc_class1').item())
        self.class_accuracies[2].append(self.trainer.callback_metrics.get('val_acc_class2').item())
        self.overall_accuracies_val.append(self.trainer.callback_metrics.get('val_acc').item())
        # print(f'Validation accuracy: {self.trainer.callback_metrics.get("val_acc").item()}')

    
    
def test():    
    data_module = preprocess.data_preprocess('test')
    model = LitResNet()
    trainer = Trainer(devices=1, accelerator="auto", log_every_n_steps=20)
    trainer.test(model, datamodule=data_module)



def fine_tuning(network):
    data_module = preprocess.data_preprocess(mode='train', batch_size=128)
    model = LitResNet(learning_rate=1e-4)
    # Training loop
    trainer = Trainer(devices=1, accelerator="auto", log_every_n_steps=1000, max_epochs=10)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    plt.figure(figsize=(10, 6))
    epochs = range(1, trainer.max_epochs + 1)
    # Plot and save training, validation, and test losses
    plt.plot(epochs, model.train_losses, label='Training Loss')
    plt.plot(epochs, model.val_losses[1:], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses')
    plt.legend()
    plt.grid()
    plt.savefig(f'{network}/losses.png')

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, model.overall_accuracies_train, label='Overall Accuracy Train')
    plt.plot(epochs, model.overall_accuracies_val[1:], label='Overall Accuracy Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Overall Training and Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(f'{network}/overall_accuracy_combined.png')
    
    # Plot and save accuracy for each class
    plt.figure(figsize=(10, 6))
    for i in range(model.num_classes):
        class_name = ''
        if i == 0:
            class_name = '9-10'
        elif i == 1:
            class_name = '10-11'
        else:
            class_name = '11-12'
        plt.plot(epochs, model.class_accuracies[i][1:], label=f'{class_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Each Class')
    plt.legend()
    plt.grid()
    plt.savefig(f'{network}/class_accuracies.png')
    # plt.show()
    
    # Assuming model.predictions contains the softmax probabilities and model.true_labels the true class indices
    predictions = np.concatenate(model.predictions)
    true_labels = np.concatenate(model.true_labels)
    # Binarize the true labels for multiclass ROC AUC
    n_classes = model.num_classes
    true_labels_binarized = label_binarize(true_labels, classes=range(n_classes))
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_binarized[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot the macro-average ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["macro"], tpr["macro"], label='Macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
            color='navy', linestyle='-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-Average ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f'{network}/Macro_Average_ROC.png')
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    fine_tuning('Resnet-101')
    # test()