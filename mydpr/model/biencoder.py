import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import esm

class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]
        tensor = tensor.contiguous()
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

def cos_sim_scores(q_vectors, ctx_vectors):
    return F.cosine_similarity(q_vectors.unsqueeze(1), ctx_vectors.unsqueeze(0), dim=-1)


def L2_dist_scores_neg(q_vectors, ctx_vectors):
    diff = q_vectors.unsqueeze(1) - ctx_vectors.unsqueeze(0)
    dists = torch.sqrt((diff * diff).sum(-1))
    return -1.0 * dists

class MyEncoder(pl.LightningModule):
    def __init__(self, proj_dim=0, bert_path=None):
        super(MyEncoder, self).__init__()
        if bert_path:
            bert1, alphabet = torch.load(bert_path[0])
            bert2, _ = torch.load(bert_path[1])
        else:
            bert1, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            bert2, _ = esm.pretrained.esm2_t12_35M_UR50D()

        self.bert1 = bert1
        self.bert2 = bert2
        
        self.alphabet = alphabet
        self.repr_layers = bert1.num_layers
        #self.num_layers = bert1.num_layers
        #repr_layers = -1
        #self.repr_layers = (repr_layers + self.num_layers + 1) % (self.num_layers + 1)
        self.recast1 = nn.Linear(480, proj_dim) if proj_dim != 0 else None
        self.recast2 = nn.Linear(480, proj_dim) if proj_dim != 0 else None
        if proj_dim !=0:
            self.dropout = nn.Dropout(0.1)

    def forward_left(self, x):
        x = self.bert1(x, repr_layers=[self.repr_layers])['representations'][self.repr_layers]
        #x[:,1:] *= 0.0
        #x = x.sum(dim=1)
        x = x[:,0]
        #x = F.normalize(x)
        if self.recast1:
            x = self.recast1(self.dropout(x))
            return F.normalize(x)
        else:
            return x

    def forward_right(self, x):
        x = self.bert2(x, repr_layers=[self.repr_layers])['representations'][self.repr_layers]
        #x[:,1:] *= 0.0
        #x = x.sum(dim=1)
        x = x[:, 0]
        #x = F.normalize(x)
        if self.recast2:
            x = self.recast2(self.dropout(x[:, 0]))
            return F.normalize(x)
        else:
            return x

    def forward(self, batch):
        seq1, seq2 = batch
        ####################################
        with torch.no_grad():
            qebd = self.forward_left(seq1)
        cebd = self.forward_right(seq2)
        ####################################
        return qebd, cebd

    def get_loss(self, ebd):
        qebd, cebd = ebd
        #qebd, cebd = qebd.half(), cebd.half()
        #qebd, cebd = qebd.float(),cebd.float()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            qebd = SyncFunction.apply(qebd)
            cebd = SyncFunction.apply(cebd)
        #####################################
        sim_mx = dot_product_scores(qebd, cebd)
        #sim_mx = L2_dist_scores_neg(qebd, cebd)
        #sim_mx = cos_sim_scores(qebd, cebd)
        label = torch.arange(sim_mx.shape[0])
        sm_score = F.log_softmax(sim_mx, dim=1)
        label = label.type_as(sm_score).long()
        return F.nll_loss(
            sm_score,
            label,
            reduction="mean"
        )
    
    def gather_all_tensor(self, ts):
        ts = ts.contiguous()
        gathered_tensor = [torch.zeros_like(ts) for _ in range(torch.distributed.get_world_size())]
        #gathered_tensor = torch.zeros((torch.distributed.get_world_size(),)+tuple(ts.shape), dtype=ts.dtype, device=ts.device)
        #gathered_tensor = gathered_tensor.contiguous()
        torch.distributed.all_gather(gathered_tensor, ts)
        gathered_tensor = torch.cat(gathered_tensor, 0)
        return gathered_tensor

    def get_acc(self, ebd):
        qebd, cebd = ebd 
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            qebd = self.gather_all_tensor(qebd)
            cebd = self.gather_all_tensor(cebd)
        sim_mx = dot_product_scores(qebd, cebd)
        #sim_mx = L2_dist_scores_neg(qebd, cebd)
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

    def qebd_step(self, batch, batch_idx, dataloader_idx=0):
        ids, seqs = batch
        ebd = self.forward_left(seqs)
        return ids, ebd.detach().cpu()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ids, seqs = batch
        ebd = self.forward_right(seqs)
        #return ids, ebd.detach().cpu()
        return ebd.detach().cpu()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=4e-5)
        return opt

    