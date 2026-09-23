import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output
    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ALSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, beta):
        ctx.save_for_backward(weight, alpha, beta)
        ctx.other = g, Qn, Qp
        eps = 1e-8
        w_q = Round.apply(torch.div(weight - beta, alpha + eps).clamp(Qn, Qp))
        w_q = w_q * alpha + beta
        return w_q
    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, beta = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        eps = 1e-8
        q_w = (weight - beta) / (alpha + eps)
        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller - bigger
        grad_alpha = ((smaller * Qn + bigger * Qp +
            between * Round.apply(q_w) - between * q_w) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_beta = ((smaller + bigger) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha, None, None, None, grad_beta


class WLSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel):
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            w_q = Round.apply(torch.div(weight, alpha)).clamp(Qn, Qp)
            w_q = w_q * alpha
            w_q = torch.transpose(w_q, 0, 1)
            w_q = w_q.contiguous().view(sizes)
        else:
            w_q = Round.apply(torch.div(weight, alpha)).clamp(Qn, Qp)
            w_q = w_q * alpha
        return w_q
    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            q_w = weight / alpha
            q_w = torch.transpose(q_w, 0, 1)
            q_w = q_w.contiguous().view(sizes)
        else:
            q_w = weight / alpha
        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller - bigger
        if per_channel:
            grad_alpha = ((smaller * Qn + bigger * Qp +
                between * Round.apply(q_w) - between * q_w) * grad_weight * g)
            grad_alpha = grad_alpha.contiguous().view(grad_alpha.size()[0], -1).sum(dim=1)
        else:
            grad_alpha = ((smaller * Qn + bigger * Qp +
                between * Round.apply(q_w) - between * q_w) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha, None, None, None, None


class CDAQActivationQuantizer(nn.Module):
    """CDAQ activation quantizer."""

    def __init__(self, a_bits, all_positive=False, batch_init=20):
        super().__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            self.Qn = 0
            self.Qp = 2 ** self.a_bits - 1
        else:
            self.Qn = -(2 ** (self.a_bits - 1))
            self.Qp = 2 ** (self.a_bits - 1) - 1
        self.mu = nn.Parameter(torch.zeros(1))
        self.log_b_minus = nn.Parameter(torch.tensor(-1.0))
        self.log_b_plus = nn.Parameter(torch.tensor(0.0))
        self.logit_alpha = nn.Parameter(torch.tensor(-4.0))
        self.register_buffer("init_state", torch.tensor(0))
        self.g = None

    def initialize_parameters(self, x):
        with torch.no_grad():
            mu_est = x.median()
            left_mask = x < mu_est
            right_mask = x >= mu_est
            if left_mask.any():
                b_minus_est = (mu_est - x[left_mask]).mean().clamp(min=1e-5)
            else:
                b_minus_est = torch.tensor(0.1, device=x.device)
            if right_mask.any():
                b_plus_est = (x[right_mask] - mu_est).mean().clamp(min=1e-5)
            else:
                b_plus_est = torch.tensor(1.0, device=x.device)
            target_log_b_minus = torch.log(b_minus_est)
            target_log_b_plus = torch.log(b_plus_est)
            if self.init_state == 0:
                self.mu.data.copy_(mu_est)
                self.log_b_minus.data.copy_(target_log_b_minus)
                self.log_b_plus.data.copy_(target_log_b_plus)
            else:
                self.mu.data = 0.9 * self.mu.data + 0.1 * mu_est
                self.log_b_minus.data = 0.9 * self.log_b_minus.data + 0.1 * target_log_b_minus
                self.log_b_plus.data = 0.9 * self.log_b_plus.data + 0.1 * target_log_b_plus

    def forward(self, x):
        if self.a_bits == 32:
            return x
        if self.training and self.init_state < self.batch_init:
            self.initialize_parameters(x.detach())
            self.init_state += 1
        b_minus = F.softplus(self.log_b_minus) + 1e-6
        b_plus = F.softplus(self.log_b_plus) + 1e-6
        alpha = torch.sigmoid(self.logit_alpha) * 0.1 + 0.0001
        log_alpha = torch.log(alpha)
        L = self.mu + b_minus * log_alpha
        U = self.mu - b_plus * log_alpha
        range_len = U - L
        range_len = range_len.clamp(min=1e-5)
        s = range_len / (self.Qp - self.Qn)
        beta = L - s * self.Qn
        if self.g is None:
            self.g = 1.0
        x_q = ALSQPlus.apply(x, s, self.g, self.Qn, self.Qp, beta)
        return x_q


class LSQPlusWeightQuantizer(nn.Module):
    """LSQ+ weight quantizer."""

    def __init__(self, w_bits, all_positive=False, per_channel=False, batch_init=20):
        super().__init__()
        self.w_bits = w_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            self.Qn = 0
            self.Qp = 2 ** w_bits - 1
        else:
            self.Qn = -2 ** (w_bits - 1)
            self.Qp = 2 ** (w_bits - 1) - 1
        self.per_channel = per_channel
        self.init_state = 0
        self.register_parameter("s", nn.Parameter(torch.ones(1), requires_grad=True))
        self.register_buffer("damp_loss", torch.tensor(0.0))
    def forward(self, weight):
        if self.w_bits == 32:
            self.damp_loss = torch.tensor(0.0, device=weight.device)
            return weight
        if self.w_bits == 1:
            raise NotImplementedError('Binary quantization is not supported!')
        if self.init_state == 0:
            self.g = 1.0 / math.sqrt(weight.numel() * self.Qp)
            self.div = 2 ** self.w_bits - 1
            if self.per_channel:
                weight_tmp = weight.detach().contiguous().view(weight.size(0), -1)
                channel_size = weight_tmp.size(1)
                mean = weight_tmp.mean(dim=1)
                if channel_size > 1:
                    var = weight_tmp.var(dim=1, unbiased=True).clamp(min=1e-12)
                else:
                    var = torch.ones_like(mean) * 1e-4
                std = torch.sqrt(var + 1e-8)
                abs_max = torch.max(
                    torch.stack([torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)]), dim=0
                )[0]
                new_scale = (abs_max / self.div).clamp(min=1e-4)
            else:
                weight_flat = weight.detach().view(-1)
                if weight_flat.numel() == 0:
                    new_scale = torch.tensor([1e-4], device=weight.device)
                elif weight_flat.numel() == 1:
                    w_val = weight_flat.item()
                    abs_max = max(abs(w_val - 3e-4), abs(w_val + 3e-4))
                    new_scale = torch.tensor([abs_max / self.div], device=weight.device).clamp(min=1e-4)
                else:
                    mean = weight_flat.mean()
                    std = weight_flat.std(unbiased=True)
                    if std == 0:
                        std = 1e-6
                    abs_max = max(abs(mean - 3 * std), abs(mean + 3 * std))
                    new_scale = torch.tensor([abs_max / self.div], device=weight.device).clamp(min=1e-4)
            self.s.data = new_scale
            self.init_state += 1
        elif self.init_state < self.batch_init:
            self.div = 2 ** self.w_bits - 1
            if self.per_channel:
                weight_tmp = weight.detach().view(weight.size(0), -1)
                channel_size = weight_tmp.size(1)
                mean = weight_tmp.mean(dim=1)
                if channel_size > 1:
                    var = weight_tmp.var(dim=1, unbiased=True).clamp(min=1e-12)
                else:
                    var = torch.ones_like(mean) * 1e-4
                std = torch.sqrt(var + 1e-8)
                abs_max = torch.max(
                    torch.stack([torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)]), dim=0
                )[0]
                new_scale = (abs_max / self.div).clamp(min=1e-4)
                self.s.data = self.s.data * 0.9 + 0.1 * new_scale
            else:
                weight_flat = weight.detach().view(-1)
                if weight_flat.numel() == 0:
                    new_scale = torch.tensor([1e-4], device=weight.device)
                elif weight_flat.numel() == 1:
                    w_val = weight_flat.item()
                    abs_max = max(abs(w_val - 3e-4), abs(w_val + 3e-4))
                    new_scale = torch.tensor([abs_max / self.div], device=weight.device).clamp(min=1e-4)
                else:
                    mean = weight_flat.mean()
                    std = weight_flat.std(unbiased=False)
                    if std == 0:
                        std = 1e-6
                    abs_max = max(abs(mean - 3 * std), abs(mean + 3 * std))
                    new_scale = torch.tensor([abs_max / self.div], device=weight.device).clamp(min=1e-4)
                self.s.data = self.s.data * 0.9 + 0.1 * new_scale
            self.init_state += 1
        elif self.init_state == self.batch_init:
            self.init_state += 1
        w_q = WLSQPlus.apply(weight, self.s, self.g, self.Qn, self.Qp, self.per_channel)
        if self.training:
            with torch.no_grad():
                if self.per_channel:
                    s_view = self.s.view(-1, 1, 1, 1)
                else:
                    s_view = self.s
                lower = self.Qn * s_view
                upper = self.Qp * s_view
            target = w_q.detach()
            weight_clipped = torch.min(torch.max(weight, lower), upper)
            self.damp_loss = F.mse_loss(weight_clipped, target, reduction='sum')
        else:
            self.damp_loss = torch.tensor(0.0, device=weight.device)
        return w_q


class QuantConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 all_positive=False,
                 per_channel=False,
                 batch_init=20):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.activation_quantizer = CDAQActivationQuantizer(a_bits=a_bits, all_positive=True, batch_init=batch_init)
        self.weight_quantizer = LSQPlusWeightQuantizer(w_bits=w_bits, all_positive=False, per_channel=per_channel, batch_init=batch_init)
        self.momentum = 0.1
        # DSC compensation
        self.register_buffer("drift_ema", torch.zeros(out_channels))
        self.register_buffer("drift_initialized", torch.zeros(1))
        self.batch_init_threshold = batch_init

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        quant_weight = self.weight_quantizer(self.weight)
        out_q = F.conv2d(
            quant_input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.training:
            if self.activation_quantizer.init_state >= self.batch_init_threshold:
                with torch.no_grad():
                    out_f = F.conv2d(input, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation, self.groups)
                    diff = out_f - out_q
                    current_drift = diff.mean(dim=(0, 2, 3))
                    current_drift = torch.clamp(current_drift, min=-0.5, max=0.5)
                    if self.drift_initialized.item() == 0:
                        self.drift_ema.data.copy_(current_drift)
                        self.drift_initialized.fill_(1)
                    else:
                        self.drift_ema.mul_(1 - self.momentum).add_(current_drift * self.momentum)
                out_q = out_q + self.drift_ema.view(1, -1, 1, 1)
        else:
            out_q = out_q + self.drift_ema.view(1, -1, 1, 1)
        return out_q


class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 a_bits=8,
                 w_bits=8,
                 all_positive=False,
                 per_channel=False,
                 batch_init=20):
        super().__init__(in_features, out_features, bias)
        self.activation_quantizer = CDAQActivationQuantizer(a_bits=a_bits, all_positive=True, batch_init=batch_init)
        self.weight_quantizer = LSQPlusWeightQuantizer(w_bits=w_bits, all_positive=False, per_channel=per_channel, batch_init=batch_init)
        self.momentum = 0.1
        # DSC compensation
        self.register_buffer("drift_ema", torch.zeros(out_features))
        self.register_buffer("drift_initialized", torch.zeros(1))
        self.batch_init_threshold = batch_init

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        quant_weight = self.weight_quantizer(self.weight)
        out_q = F.linear(quant_input, quant_weight, self.bias)
        if self.training:
            if self.activation_quantizer.init_state >= self.batch_init_threshold:
                with torch.no_grad():
                    out_f = F.linear(input, self.weight, self.bias)
                    diff = out_f - out_q
                    reduce_dims = list(range(diff.dim() - 1))
                    current_drift = diff.mean(dim=reduce_dims)
                    current_drift = torch.clamp(current_drift, min=-0.5, max=0.5)
                    if self.drift_initialized.item() == 0:
                        self.drift_ema.data.copy_(current_drift)
                        self.drift_initialized.fill_(1)
                    else:
                        self.drift_ema.mul_(1 - self.momentum).add_(current_drift * self.momentum)
                out_q = out_q + self.drift_ema
        else:
            out_q = out_q + self.drift_ema
        return out_q

    def fold_drift_to_bias(self):
        if self.bias is None:
            self.bias = nn.Parameter(self.drift_ema.clone())
        else:
            self.bias.data.add_(self.drift_ema)
        self.drift_ema.zero_()


def count_conv_layers(module):
    count = 0
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            count += 1
    return count


def add_quant_op(
    module,
    layer_counter,
    total_convs,
    quant_first=True,
    quant_last=True,
    default_a_bits=8,
    default_w_bits=8,
    first_a_bits=8,
    first_w_bits=8,
    last_a_bits=8,
    last_w_bits=8,
    all_positive=False,
    per_channel=False,
    batch_init=20
):
    for name, child in module.named_children():
        if "model_up" in name:
            print(f"Skip quantizing {name}")
            continue
        if isinstance(child, nn.Conv2d):
            layer_counter[0] += 1
            current_layer = layer_counter[0]
            is_first = (current_layer == 1)
            is_last = (current_layer == total_convs) or (current_layer == total_convs - 1) or (current_layer == total_convs - 2)
            should_quantize = True
            if is_first and not quant_first:
                should_quantize = False
            elif is_last and not quant_last:
                should_quantize = False
            if should_quantize:
                if is_first:
                    a_bits, w_bits = first_a_bits, first_w_bits
                elif is_last:
                    a_bits, w_bits = last_a_bits, last_w_bits
                else:
                    a_bits, w_bits = default_a_bits, default_w_bits
                quant_conv = QuantConv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    padding_mode=child.padding_mode,
                    a_bits=a_bits,
                    w_bits=w_bits,
                    all_positive=all_positive,
                    per_channel=per_channel,
                    batch_init=batch_init
                )
                quant_conv.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    quant_conv.bias.data = child.bias.data.clone()
                module._modules[name] = quant_conv
        elif isinstance(child, nn.Linear):
            quant_linear = QuantLinear(
                child.in_features,
                child.out_features,
                bias=(child.bias is not None),
                a_bits=default_a_bits,
                w_bits=default_w_bits,
                all_positive=all_positive,
                per_channel=per_channel,
                batch_init=batch_init
            )
            quant_linear.weight.data = child.weight.data.clone()
            if child.bias is not None:
                quant_linear.bias.data = child.bias.data.clone()
            module._modules[name] = quant_linear
        else:
            add_quant_op(
                child, layer_counter, total_convs,
                quant_first, quant_last,
                default_a_bits, default_w_bits,
                first_a_bits, first_w_bits,
                last_a_bits, last_w_bits,
                all_positive, per_channel, batch_init
            )


def prepare(
    model,
    inplace=False,
    quant_first=True,
    quant_last=True,
    default_a_bits=8,
    default_w_bits=8,
    first_a_bits=8,
    first_w_bits=8,
    last_a_bits=8,
    last_w_bits=8,
    all_positive=False,
    per_channel=False,
    batch_init=20, layerid=None
):
    if not inplace:
        model = copy.deepcopy(model)
    total_convs = count_conv_layers(model)
    print(f"Total Conv2d layers: {total_convs}")
    layer_counter = [0]
    add_quant_op(
        model, layer_counter, total_convs,
        quant_first=quant_first,
        quant_last=quant_last,
        default_a_bits=default_a_bits,
        default_w_bits=default_w_bits,
        first_a_bits=first_a_bits,
        first_w_bits=first_w_bits,
        last_a_bits=last_a_bits,
        last_w_bits=last_w_bits,
        all_positive=all_positive,
        per_channel=per_channel,
        batch_init=batch_init
    )
    return model
