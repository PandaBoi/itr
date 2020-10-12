from time import time
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger




def preproc_data():
    from data import split_data
    split_data('../data/hin-eng/hin.txt', '../data/hin-eng')


def init_seed():
    seed_val = 42
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


from data import IndicDataset, PadSequence
import model as M



class LightModule(pl.LightningModule):
 
  def __init__(self,config):
    super(LightModule, self).__init__()
    self.hparam = config
    init_seed()
    preproc_data()
    
    self.model, self.tokenizers = M.build_model(config)
    self.pad_sequence = PadSequence(self.tokenizers.src.pad_token_id, self.tokenizers.tgt.pad_token_id)
    print('init success')
  
  def train_dataloader(self):
   
    return DataLoader(IndicDataset(self.tokenizers.src, self.tokenizers.tgt, self.hparam.data, True), 
                                  batch_size=self.hparam.batch_size, 
                                  shuffle=False, 
                                  collate_fn=self.pad_sequence)
  
  def val_dataloader(self):

    return DataLoader(IndicDataset(self.tokenizers.src, self.tokenizers.tgt, self.hparam.data, False), 
                                  batch_size=self.hparam.eval_size, 
                                  shuffle=False, 
                                  collate_fn=self.pad_sequence)

  def forward(self, x):
    return self.model(x)
  
  def training_step(self, batch, batch_idx):

    source, target = batch
    loss, logits = self.model(source, target)
    logs = {'train_loss':loss}
    # self.logger.log_metrics(logs)
    return loss
  
  def training_epoch_end(self, training_step_op):

    train_avg_loss = torch.stack([x['loss'] for x in training_step_op]).mean()
    logs = {'train_epoch_loss': train_avg_loss}
    self.logger.experiment.add_scalar('Loss_Train',train_avg_loss,self.current_epoch)
    
  
  def validation_step(self, batch, batch_idx):

    # print('VAL_step')
    source, target = batch
    loss, logits = self.model(source, target)
    pred_flat = torch.argmax(logits, axis=2).flatten()
    labels_flat = target.flatten()
    correct = (pred_flat == labels_flat).sum().float()
    total = len(pred_flat)
    logs = {'val_loss':loss,'correct': correct}
    # self.logger.log_metrics(logs)
    return {'loss':loss, 'correct': correct, 'total' : total}
  
  def validation_epoch_end(self, val_step_op):

    # print('VAL_end')

    val_avg_loss = torch.stack([x['loss'] for x in val_step_op]).mean()
    acc = torch.stack([x['correct'] for x in val_step_op]).sum()
    total_arr = [x['total'] for x in val_step_op]
    total = sum(total_arr)
    acc /= total
    self.logger.experiment.add_scalar('Acc_Test',acc,self.current_epoch)
    self.logger.experiment.add_scalar('Loss_Test',val_avg_loss,self.current_epoch)
    
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.train_dataloader()), eta_min=self.hparam.lr)

    return [optimizer],[scheduler]
  

  




from config import replace, preEnc, preEncDec
def main():
    rconf = preEncDec
    model = LightModule(rconf)
    
    logger = TensorBoardLogger(
              rconf.log_dir,
              'tb_logs',
              
    )

    ckpt_callback = ModelCheckpoint(
                    monitor = 'Acc_Test',
                    filepath = str(rconf.output_dir) + '/model_weights' ,
                    mode = 'max',
                    save_top_k = 1,
                    save_last = True
    )

    trainer = Trainer(
              max_epochs = rconf.epochs,
              default_root_dir = rconf.log_dir,
              gpus =  [0],
              logger = logger,
              check_val_every_n_epoch = 1,
              checkpoint_callback = ckpt_callback
    )
    trainer.fit(model)
    

if __name__ == '__main__':
    #preproc_data()
    main()








