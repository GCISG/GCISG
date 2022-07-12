import torch
from torch import nn
import torch.nn.functional as F


class CILoss(nn.Module):
    def __init__(
        self,
        scale,
        temperature_q,
        temperature_k,
        stages,
        embedding_dim,
        queue_size,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.temperature_q = temperature_q
        self.temperature_k = temperature_k
        self.stages = stages
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.scale = scale
        self.ddp = True

        self.initialized = False

        for stage in self.stages:
            self.register_buffer(
                f"queue_{stage}",
                torch.randn(self.embedding_dim, self.queue_size),
            )
            self.register_buffer(f"queue_ptr_{stage}", torch.zeros(1, dtype=torch.long))
            setattr(
                self,
                f"queue_{stage}",
                nn.functional.normalize(getattr(self, f"queue_{stage}"), dim=0),
            )

        self.criterion = nn.CrossEntropyLoss()

    def on_forward(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, stage):
        # gather keys before updating queue
        if self.ddp:
            keys = concat_all_gather(keys)

        queue = getattr(self, f"queue_{stage}")
        queue_ptr = getattr(self, f"queue_ptr_{stage}")

        self._dequeue_and_euqueue_stage(keys, queue, queue_ptr)

    @torch.no_grad()
    def _dequeue_and_euqueue_stage(self, keys, queue, queue_ptr):
        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        # assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        queue_ptr[0] = ptr

    def forward(self, q, k, *args, **kwargs):
        if not self.initialized:
            for stage in self.stages:
                setattr(
                    self,
                    f"queue_{stage}",
                    getattr(self, f"queue_{stage}").to(q.device),
                )
            self.initialized = True

        stage = kwargs["stage"]

        p1 = torch.einsum(
            "nc,ck->nk", [k, getattr(self, f"queue_{stage}").clone().detach()]
        )
        p2 = torch.einsum(
            "nc,ck->nk", [q, getattr(self, f"queue_{stage}").clone().detach()]
        )

        loss = -torch.sum(
            F.softmax(p1.detach() / self.temperature_k, dim=1)
            * F.log_softmax(p2 / self.temperature_q, dim=1),
            dim=1,
        ).mean()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, stage)

        return self.scale * loss


# TODO: Remove duplicate
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
