import torch.nn as nn
import torch
from inspect import signature
from collections import OrderedDict
from einops import repeat
import torch.nn.functional as F


class CLIP_Transform(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Project to a lower resolution feature map
        self.fc = nn.Linear(hidden_dim, 256*8*8)
        self.upsample = nn.Upsample((128, 128), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.fc(x)
        # Reshape to [bz, 256, 8, 8]
        x = x.view(-1, 256, 8, 8)
        # Upsample to [bz, 256, 128, 128]
        x = self.upsample(x)
        return x
    

class AbstractModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.compiled = False

    # Compile module and assign optimizer + params
    def compile(self, optimizer=None, **kwargs):

        if optimizer is not None:
            self.optimizer_class = optimizer
            self.optimizer_kwargs = kwargs
            self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        else:
            self.optimizer = None

        self.compiled = True
        self.to(DEVICE)

    # Predict scores from a batch of data
    def predict_on_batch(self, data):

        self.eval()
        with torch.no_grad():
            return self.forward(data)

    # Fit (make one optimizer step) on a batch of data
    def fit_on_batch(self, data, target, loss_fn=None, train=True):
        loss_fn = loss_fn or self.loss

        self.zero_grad()
        self.optimizer.zero_grad()

        self.train(train)

        self.zero_grad()
        self.optimizer.zero_grad()
        pred = self.forward(data)
        if isinstance(target, list):
            target = tuple(t.to(pred.device) for t in target)
        else: target = target.to(pred.device)

        if len(signature(loss_fn).parameters) > 2:
            loss, metrics = loss_fn(pred, target, data.to(pred.device))
        else:
            loss, metrics = loss_fn(pred, target)

        if train:
            loss.backward()
            self.optimizer.step()
            self.zero_grad()
            self.optimizer.zero_grad()

        return pred, loss, metrics

    # Make one optimizer step w.r.t a loss
    def step(self, loss, train=True):

        # self.zero_grad()
        # self.optimizer.zero_grad()
        self.train(train)
        self.zero_grad()
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()
        self.zero_grad()
        self.optimizer.zero_grad()

    @classmethod
    def load(cls, weights_file=None):
        model = cls()
        if weights_file is not None:
            data = torch.load(weights_file)
            # hack for models saved with optimizers
            if "optimizer" in data: data = data["state_dict"]
            model.load_state_dict(data)
        return model

    def load_weights(self, weights_file, backward_compatible=False):
        data = torch.load(weights_file)
        if backward_compatible:
            data = {'parallel_apply.module.'+k:v for k,v in data.items()}
        self.load_state_dict(data)

    def save(self, weights_file):
        torch.save(self.state_dict(), weights_file)

    # Subclasses: override for custom loss + forward functions
    def loss(self, pred, target):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


class TrainableModel(AbstractModel):
    def __init__(self):
        super().__init__()

    # Fit on generator for one epoch
    def _process_data(self, datagen, loss_fn=None, train=True, logger=None):

        self.train(train)
        out = []
        for data in datagen:
            batch, y = data[0], data[1:]
            if len(y) == 1: y = y[0]
            y_pred, loss, metric_data = self.fit_on_batch(batch, y, loss_fn=loss_fn, train=train)
            if logger is not None:
                logger.update("loss", float(loss))
            yield ((batch.detach(), y_pred.detach(), y, float(loss), metric_data))

    def fit(self, datagen, loss_fn=None, logger=None):
        for x in self._process_data(datagen, loss_fn=loss_fn, train=train, logger=logger):
            pass

    def fit_with_data(self, datagen, loss_fn=None, logger=None):
        images, preds, targets, losses, metrics = zip(
            *self._process_data(datagen, loss_fn=loss_fn, train=True, logger=logger)
        )
        images, preds, targets = torch.cat(images, dim=0), torch.cat(preds, dim=0), torch.cat(targets, dim=0)
        metrics = zip(*metrics)
        return images, preds, targets, losses, metrics

    def fit_with_metrics(self, datagen, loss_fn=None, logger=None):
        metrics = [
            metrics
            for _, _, _, _, metrics in self._process_data(
                datagen, loss_fn=loss_fn, train=True, logger=logger
            )
        ]
        return list(zip(*metrics))

    def predict_with_data(self, datagen, loss_fn=None, logger=None):
        images, preds, targets, losses, metrics = zip(
            *self._process_data(datagen, loss_fn=loss_fn, train=False, logger=logger)
        )
        images, preds, targets = torch.cat(images, dim=0), torch.cat(preds, dim=0), torch.cat(targets, dim=0)
        images, preds, targets = images.cpu(), preds.cpu(), targets.cpu()
        # preds = torch.cat(preds, dim=0)
        metrics = zip(*metrics)
        return images, preds, targets, losses, metrics

    def predict_with_metrics(self, datagen, loss_fn=None, logger=None):
        metrics = [
            metrics
            for _, _, _, _, metrics in self._process_data(
                datagen, loss_fn=loss_fn, train=False, logger=logger
            )
        ]
        return list(zip(*metrics))

    def predict(self, datagen):
        preds = [self.predict_on_batch(x) for x in datagen]
        preds = torch.cat(preds, dim=0)
        return preds


class DataParallelModel(TrainableModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.parallel_apply = nn.DataParallel(*args, **kwargs)

    def forward(self, x):
        return self.parallel_apply(x)

    def loss(self, x, preds):
        return self.parallel_apply.module.loss(x, preds)

    @property
    def module(self):
        return self.parallel_apply.module

    @classmethod
    def load(cls, model=TrainableModel(), weights_file=None):
        model = cls(model)
        if weights_file is not None:
            data = torch.load(weights_file, map_location=lambda storage, loc: storage)
            # hack for models saved with optimizers
            if "optimizer" in data: data = data["state_dict"]
            model.load_state_dict(data)
        return model

class WrapperModel(TrainableModel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def loss(self, x, preds):
        raise NotImplementedError()

    def __getitem__(self, i):
        return self.model[i]

    @property
    def module(self):
        return self.model

class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))
    
class LayerNorm(nn.LayerNorm):
    # We always use float32 for the LayerNorm for stable training
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight.to(torch.float32), self.bias.to(torch.float32), self.eps)
        return ret.type(orig_type)

class PerceiverAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)

        self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("sq_relu", SquaredReLU()),
                ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))

        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.ln_ff = LayerNorm(d_model)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        return self.attn(q, kv, kv, need_weights=False)[0]

    def forward(self, x: torch.Tensor, latents: torch.Tensor):
        latents = latents + self.attention(q=self.ln_1(latents), kv=torch.cat([self.ln_1(latents), self.ln_2(x)], dim=0))
        latents = latents + self.mlp(self.ln_ff(latents))
        return latents


class PerceiverResampler(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, num_latents: int):
        super().__init__()
        scale = width ** -0.5
        self.latents = nn.Parameter(scale * torch.randn(num_latents, width))
        self.perceiver_blocks = nn.Sequential(*[PerceiverAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x_f: torch.Tensor):
        x = repeat(self.latents, 'l d -> l b d', b=x_f.shape[1])

        for p_block in self.perceiver_blocks:
            x = p_block(x_f, x)

        return x  # num_latents, batch_size, output_dim


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embedding_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embedding_dim, H', W')
        x = x.flatten(2)  # (B, embedding_dim, H'*W')
        x = x.transpose(1, 2)  # (B, H'*W', embedding_dim)
        return x