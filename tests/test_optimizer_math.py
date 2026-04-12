import torch

from infinity.optimizer import AdamWOptimizer, ParameterState


def test_custom_adamw_matches_torch_adamw():
    torch.manual_seed(0)

    base = torch.randn(8, 8, dtype=torch.float32)
    grads = [torch.randn_like(base), torch.randn_like(base)]

    custom_param = ParameterState(base.clone(), name="test_param")
    custom_opt = AdamWOptimizer(
        [custom_param],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        max_grad_norm=None,
    )

    torch_param = torch.nn.Parameter(base.clone())
    torch_opt = torch.optim.AdamW(
        [torch_param],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    for grad in grads:
        custom_param.grad.copy_(grad)
        torch_param.grad = grad.clone()

        custom_opt.step()
        torch_opt.step()
        torch_opt.zero_grad(set_to_none=True)

        assert torch.allclose(
            custom_param.master,
            torch_param.detach(),
            atol=1e-7,
            rtol=1e-6,
        )
