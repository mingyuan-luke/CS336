import torch
import torch.nn as nn

class AdamW(nn.Module):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__()
        # self.optimizer = torch.optim.AdamW(
        #     parameters,
        #     lr=lr,
        #     betas=betas,
        #     eps=eps,
        #     weight_decay=weight_decay
        # )

        # torch.optim.Optimizer
        self.optimizer = torch.optim.Optimizer(
            parameters,
            defaults={
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay
            }
        )
        self.state = {}  # to store optimizer state like moments for AdamW


    def step(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state.setdefault(p, {})
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data) # 一阶矩
                    state['exp_avg_sq'] = torch.zeros_like(p.data) # 二阶矩

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = self.optimizer.defaults['betas']

                state['step'] += 1

                grad = p.grad.data

                # Compute adjusted lr for iteration 𝑡
                adjusted_lr = self.optimizer.defaults['lr'] * (1 - beta2 ** state['step']) ** 0.5 / (1 - beta1 ** state['step'])

                #Apply weight decay, θ ← θ − αλθ, decoupled weight decay
                if self.optimizer.defaults['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-self.optimizer.defaults['lr'] * self.optimizer.defaults['weight_decay'])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Apply moment-adjusted weight updates
                denom = exp_avg_sq.sqrt().add_(self.optimizer.defaults['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-adjusted_lr)


    def zero_grad(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()