import os

import torch
import torch.nn as nn
from torch.nn import functional as F

# from training.zero_shot import accuracy 

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)

            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


def my_all_gather(q):
    ws = int(os.environ['WORLD_SIZE'])
    local_size = torch.tensor(q.shape[0], device=q.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(all_sizes)

    size_diff = max_size - q.shape[0]
    if size_diff > 0:
        padding = torch.zeros(size_diff, q.shape[1], device=q.device, dtype=q.dtype)
        q = torch.cat((q, padding))

    all_qs_padded = [torch.zeros_like(q) for _ in range(ws)]
    dist.all_gather(all_qs_padded, q)
    output_l = []
    for q, size in zip(all_qs_padded, all_sizes):
        output_l.append(q[:size])
    return output_l


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits, cum_rank_size=None) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                offset = num_logits * self.rank if cum_rank_size is None else cum_rank_size[self.rank]
                labels = labels + offset
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, all_image_features=None, all_text_features=None,
                   mask=None):
        if self.world_size > 1:
            if all_image_features is None and all_text_features is None:
                all_image_features, all_text_features = gather_features(
                    image_features, text_features,
                    self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                if mask is None:
                    mask = torch.zeros(1, dtype=image_features.dtype, device=image_features.device)
                logits_per_image = logit_scale * image_features @ all_text_features.T + mask
                logits_per_text = logit_scale * text_features @ all_image_features.T + mask
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, mask=None, all_image_features=None,
                all_text_features=None, cum_rank_size=None, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale,
                                                            all_image_features=all_image_features,
                                                            all_text_features=all_text_features, mask=mask)

        labels = self.get_ground_truth(device, logits_per_image.shape[0], cum_rank_size=cum_rank_size)

        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):

        clip_loss = torch.tensor(0)

        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
                                   F.cross_entropy(logits_per_image, labels) +
                                   F.cross_entropy(logits_per_text, labels)
                           ) / 2

        distill_loss = (
                               self.dist_loss(dist_logits_per_image, logits_per_image) +
                               self.dist_loss(dist_logits_per_text, logits_per_text)
                       ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
               NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss


class ZSLoss(nn.Module):
    def __init__(self, use_dense_loss=False, args=None, **kwargs):
        super().__init__()
        self.use_dense_loss = use_dense_loss
        self.cliploss = ClipLoss(**kwargs)
        kwargs['cache_labels'] = False
        self.cliploss2 = ClipLoss(**kwargs)
        self.args = args
        self.n_classes = args.n_classes
        self.class_counts = torch.zeros(self.n_classes, device='cuda', requires_grad=False)
        self.lambda_loss = args.lambda_loss

    def forward(self, image_features, text_features, logit_scale, cls_embeds_txt, cls_embeds_visual, logit_scale_local,
                classification_weights_ema, concepts_labels, output_dict=False):
        all_image_features, all_text_features, all_cls_embeds_txt, all_concepts_labels, cum_rank_size, all_classification_weights_ema = self.gather_features(
            image_features, text_features, cls_embeds_txt, concepts_labels, classification_weights_ema)

        cliploss = self.cliploss(image_features, text_features, logit_scale, all_image_features=all_image_features,
                                 all_text_features=all_text_features, output_dict=False)
        if self.use_dense_loss:
            # Compute the classification weights
            all_concepts_labels = all_concepts_labels.to(torch.long)
            n, _ = all_cls_embeds_txt.shape

            dummy_counts = torch.ones(n, device=cls_embeds_txt.device, dtype=cls_embeds_txt.dtype)
            batch_class_counts = torch.zeros(self.n_classes, device=cls_embeds_txt.device, dtype=cls_embeds_txt.dtype)
            batch_class_counts = torch.scatter_reduce(
                input=batch_class_counts,
                dim=0,
                index=all_concepts_labels,
                src=dummy_counts,
                reduce='sum'
            )
            self.class_counts += batch_class_counts
            n_non_zero_classes = (batch_class_counts > 0).sum()
            all_n_non_zero_classes = (self.class_counts > 0).sum()

            # Project
            classification_weights_ema = classification_weights_ema[:-1]
            logits = logit_scale_local * torch.einsum('c d, b d -> b c', classification_weights_ema, cls_embeds_visual)

            # Infer the labels
            with torch.no_grad():
                logits_ = logits.detach().clone().permute(1, 0)
                logits_[batch_class_counts == 0] = - float('inf')
                patch_predictions = logits_.argmax(dim=0)

                # logging
                softmax_ = torch.softmax(logits, dim=-1)
                s_min = softmax_.min()
                s_max = softmax_.max()

            # Compute the accuracy
            with torch.no_grad():  # Probably not needed
                valid_predictions = patch_predictions == concepts_labels
            acc = valid_predictions.float().mean() * 100

            # Mask
            logits[:, batch_class_counts == 0] = - float('inf')

            # Compute the dense loss
            if logits.numel() == 0:
                dense_loss = torch.tensor(0., device=image_features.device)
            else:
                dense_loss = F.cross_entropy(logits, patch_predictions)
        else:
            dense_loss = torch.tensor(0., device=image_features.device)
            acc = torch.tensor(0., device=image_features.device)
            n_non_zero_classes = torch.tensor(0., device=image_features.device)
            all_n_non_zero_classes = torch.tensor(0., device=image_features.device)
            s_min = torch.tensor(0., device=image_features.device)
            s_max = torch.tensor(0., device=image_features.device)

        # Re-weight the losses
        dense_loss *= self.lambda_loss
        if output_dict:
            return ({"cliploss": cliploss, "ccloss": dense_loss}, all_classification_weights_ema, acc.detach(),
                    n_non_zero_classes, all_n_non_zero_classes, s_min, s_max)
        return (
        cliploss + dense_loss, all_classification_weights_ema, acc.detach(), n_non_zero_classes, all_n_non_zero_classes,
        s_min, s_max)

    def gather_features(self, image_features, text_features, cls_embeds_txt, nounsID, classification_weights_ema):
        if self.cliploss.world_size == 1:
            return image_features, text_features, cls_embeds_txt, nounsID, None, classification_weights_ema

        all_nounsID = None
        cum_rank_size = None

        B = image_features.shape[0]
        # Do custom gather
        if self.args.n_dense_cls > 0:
            # max_size = self.args.n_dense_cls * B
            max_size = 10 * B
            padding_size = max_size - cls_embeds_txt.shape[0]

            to_gather = torch.cat([
                classification_weights_ema,
                image_features,
                text_features,
                cls_embeds_txt,
                torch.zeros(padding_size, image_features.shape[-1], dtype=image_features.dtype,
                            device=image_features.device),
            ], dim=0)

            to_gather_nouns = torch.cat([
                self.class_counts,
                nounsID,
                -torch.ones(padding_size, dtype=nounsID.dtype, device=nounsID.device)
            ], dim=0)

            if self.cliploss.gather_with_grad:
                gathered = torch.distributed.nn.all_gather(to_gather)
                gathered_nouns = torch.distributed.nn.all_gather(to_gather_nouns)
                all_class_counts, all_nounsID = list(
                    zip(*[(g_n[:self.n_classes], g_n[self.n_classes:][g_n[self.n_classes:] != -1]) for g_n in
                          gathered_nouns]))
                all_nounsID = torch.cat(all_nounsID, dim=0)
                all_class_counts = torch.stack(all_class_counts, dim=0)
            else:
                gathered = [torch.zeros_like(to_gather) for _ in range(self.cliploss.world_size)]
                dist.all_gather(gathered, to_gather)
                gathered[self.cliploss.rank] = to_gather
                raise NotImplementedError

            lst = []
            cum_rank_size = [0]
            all_classification_weights_ema = []
            for tensor in gathered:
                all_classification_weights_ema.append(tensor[:self.n_classes])
                lean = tensor[self.n_classes:]
                lean = lean[~torch.all(lean == 0.0, dim=-1)]
                splitted = torch.tensor_split(lean, (B, 2 * B), dim=0)
                lst.append(splitted)
                cum_rank_size.append(splitted[1].shape[0])

            cum_rank_size = torch.Tensor(cum_rank_size[:-1]).to(torch.int32).cumsum(dim=0)

            all_image_features, all_text_features, all_cls_embeds_txt = \
                [torch.cat(i, dim=0) for i in zip(*lst)]

            # Average the classification weights
            all_classification_weights_ema = torch.stack(all_classification_weights_ema, dim=0)
            all_classification_weights_ema = torch.einsum('g n d, g n -> n d', all_classification_weights_ema,
                                                          all_class_counts)
            all_classification_weights_ema = F.normalize(all_classification_weights_ema, dim=-1, p=2)

            return all_image_features, all_text_features, all_cls_embeds_txt, all_nounsID, cum_rank_size, all_classification_weights_ema

        # Do standard gather
        else:
            if self.cliploss.gather_with_grad:
                all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
                all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            else:
                gathered_image_features = [torch.zeros_like(image_features) for _ in range(self.cliploss.world_size)]
                gathered_text_features = [torch.zeros_like(text_features) for _ in range(self.cliploss.world_size)]
                dist.all_gather(gathered_image_features, image_features)
                dist.all_gather(gathered_text_features, text_features)

                gathered_image_features[self.cliploss.rank] = image_features
                gathered_text_features[self.cliploss.rank] = text_features

                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)

            return all_image_features, all_text_features, None, all_nounsID, cum_rank_size, None
