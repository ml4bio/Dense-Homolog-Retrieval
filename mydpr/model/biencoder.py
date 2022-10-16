import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import esm

class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


def dot_product_scores(q_vectors, ctx_vectors):
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    return torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))


class MyEncoder(pl.LightningModule):
    def __init__(self, proj_dim=0):
        super(MyEncoder, self).__init__()
        bert, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
        self.bert = bert 
        self.alphabet = alphabet
        self.num_layers = bert.num_layers
        repr_layers = -1
        self.repr_layers = (repr_layers + self.num_layers + 1) % (self.num_layers + 1)
        self.recast = nn.Linear(768, proj_dim) if proj_dim != 0 else None

    def forward_once(self, x):
        x = self.bert(x, repr_layers=[self.repr_layers])['representations'][self.repr_layers]
        x[:,1:] *= 0
        x = x.sum(dim=1)
        #x = x[:,0]
        if self.recast :
            x = self.recast(x)
        return x

    def forward(self, batch):
        seq1, seq2 = batch
        ####################################
        qebd = self.forward_once(seq1)
        cebd = self.forward_once(seq2)
        #qebd = F.normalize(self.forward_once(seq1))
        #cebd = F.normalize(self.forward_once(seq2))
        ####################################
        return qebd, cebd

    def get_loss(self, ebd):
        qebd, cebd = ebd
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            qebd = SyncFunction.apply(qebd)
            cebd = SyncFunction.apply(cebd)
        #####################################
        sim_mx = dot_product_scores(qebd, cebd)
        label = torch.arange(sim_mx.shape[0])
        sm_score = F.log_softmax(sim_mx, dim=1)
        label = label.type_as(sm_score).long()
        return F.nll_loss(
            sm_score,
            label,
            reduction="mean"
        )
    
    def gather_all_tensor(self, ts):
        gathered_tensor = [torch.zeros_like(ts) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_tensor, ts)
        gathered_tensor = torch.cat(gathered_tensor, 0)
        return gathered_tensor

    def get_acc(self, ebd):
        qebd, cebd = ebd 
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            qebd = self.gather_all_tensor(qebd)
            cebd = self.gather_all_tensor(cebd)
        sim_mx = dot_product_scores(qebd, cebd)
        label = torch.arange(sim_mx.shape[0], dtype=torch.long)
        sm_score = F.log_softmax(sim_mx, dim=1)
        max_score, max_idxs = torch.max(sm_score, 1)
        correct_predictions_count = (
            max_idxs == label.to(sm_score.device)
        ).sum()
        return correct_predictions_count, sim_mx.shape[0]

    def training_step(self, batch, batch_idx):
        ebd = self.forward(batch)
        loss = self.get_loss(ebd)
        with torch.no_grad():
            hit, tot = self.get_acc(ebd)
            acc = hit / tot
        values = {
            'train_loss': loss,
            'train_acc' : acc,
            'total' : tot,
        }
        self.log_dict(values, on_step=True, on_epoch=True, sync_dist=False, rank_zero_only=True)
        #sch = self.lr_schedulers()
        #sch.step()
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        ebd = self(batch)
        val_loss = self.get_loss(ebd)
        hit, tot  = self.get_acc(ebd)
        val_acc = hit.cpu() / tot
        values = {
            'val_loss': val_loss,
            'val_acc' : val_acc,
        }
        self.log_dict(values, on_step=True, on_epoch=True, sync_dist=True, rank_zero_only=True)
        return values

    def test_step(self, batch, batch_idx):
        ebd = self(batch)
        test_loss = self.get_loss(ebd)
        hit, tot  = self.get_acc(ebd)
        test_acc  = hit.cpu() / tot
        values = {
            'test_loss': test_loss,
            'test_acc' : test_acc,
        }
        self.log_dict(values, on_step=True, on_epoch=True, sync_dist=True, rank_zero_only=True)
        return values

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ids, seqs = batch
        ebd = self.forward_once(seqs)
        return ids, ebd.detach().cpu()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=4e-5)
        #sch = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=80, gamma=0.85)
        return opt
        '''
        sch = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=80, gamma=0.85)
        return {
            'optimizer': opt,
            'lr_scheduler': sch,
        }
        '''

    