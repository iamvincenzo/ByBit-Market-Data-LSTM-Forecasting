import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model_cfg import config
from PlottingUtils import Visualizer

class Solver():
    """ Initialize configuration. """
    def __init__(self, model, train_dataloader, val_dataloader):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Getting parameters 
        ############################
        self.device = config["training"]["device"]
        self.num_epoch = config["training"]["num_epoch"]
        learning_rate = config["training"]["learning_rate"]       
        scheduler_step_size =  config["training"]["scheduler_step_size"]
        ###########################

        # define optimizer, scheduler and loss function
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, 
                                    betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, 
                                                   gamma=0.1)
        
        self.vz = Visualizer()
        
    def run_epoch(self, epoch, is_training=False):
        epoch_loss = 0

        if is_training:
            print(f'\nTraining iteration | '
                  f'Epoch[{epoch + 1}/{self.num_epoch}]\n')
            self.model.train()
            dataloader = self.train_dataloader
        else:
            print(f'\nEvaluation iteration | ' 
                  f'Epoch [{epoch + 1}/{self.num_epoch}]\n')
            self.model.eval()
            dataloader = self.val_dataloader

        from tqdm import tqdm

        loop = tqdm(enumerate(dataloader),
                    total=len(dataloader), 
                    leave=True)

        for batch_idx, (X_batch, y_batch) in loop:
            if is_training:
                self.optimizer.zero_grad()

            batchsize = X_batch.shape[0]

            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            out = self.model(X_batch)

            """ Le stringhe grad_fn=<AddmmBackward0> e grad_fn=<SelectBackward0> sono due esempi di funzioni 
                gradiente (grad_fn) associate a tensori in PyTorch, che rappresentano la computazione che ha 
                portato alla creazione di quel tensore. La differenza tra le due stringhe sta nel tipo di 
                operazione che è stata eseguita per creare il tensore. In particolare:
                    - grad_fn=<AddmmBackward0> indica che il tensore è stato creato tramite una moltiplicazione 
                        tra due tensori, seguita da una somma. Questo tipo di operazione è comunemente utilizzato 
                        in reti neurali per eseguire la moltiplicazione tra i pesi e le feature di input, seguita 
                        dalla somma del bias.

                    - grad_fn=<SelectBackward0> indica che il tensore è stato creato tramite una selezione di un 
                        sottoinsieme di elementi di un altro tensore. Questo tipo di operazione è comunemente utilizzato
                        in reti neurali per selezionare un sottoinsieme di feature di input o di attivazioni di un layer 
                        per essere passati al successivo.

                Quale usare?
                    - Se si utilizzano tensori di grandi dimensioni, l'operazione di moltiplicazione tra i pesi e 
                        le feature di input (che genera la funzione gradiente grad_fn=<AddmmBackward0>) può richiedere 
                        più tempo di elaborazione rispetto alla selezione di un sottoinsieme di elementi di un altro 
                        tensore (che genera la funzione gradiente grad_fn=<SelectBackward0>);

                    - Se si utilizzano tensori di piccole dimensioni, l'operazione di selezione potrebbe richiedere meno 
                        tempo di elaborazione rispetto alla moltiplicazione.

                    - Inoltre, l'efficienza delle funzioni gradiente può variare a seconda del tipo di hardware utilizzato 
                        per l'esecuzione dell'operazione. Ad esempio, alcune operazioni possono essere eseguite più velocemente 
                        su schede grafiche (GPU) rispetto alla CPU.
            """ 
            
            """ (grad_fn=<AddmmBackward0>) 
            loss = self.criterion(out.contiguous(), y_batch.contiguous())
            """

            """ grad_fn=<SelectBackward0> """ 
            loss = self.criterion(out.contiguous()[:, -1], y_batch.contiguous()[:, -1])

            if is_training:
                loss.backward()
                self.optimizer.step()

            epoch_loss += (loss.detach().item() / batchsize)

        lr = self.scheduler.get_last_lr()[0]

        return epoch_loss, lr

    def train_eval(self):
        train_losses = []
        val_losses = []
        for epoch in range(self.num_epoch):
            loss_train, lr_train = self.run_epoch(epoch, is_training=True)
            loss_val, lr_val = self.run_epoch(epoch)
            self.scheduler.step()

            print('\nEpoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
                .format(epoch + 1, self.num_epoch, loss_train, loss_val, lr_train))

            train_losses.append(loss_train)
            val_losses.append(loss_val)
            
        self.vz.plot_loss(train_losses, val_losses)

    def make_pred(self):
        pass