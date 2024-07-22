import torch


def attention_rollout(attentions):
    """Computes attention rollout from the given list of attention matrices.
    https://arxiv.org/abs/2005.00928
    """
    rollout = attentions[0]
    for A in attentions[1:]:
        rollout = torch.matmul(
            0.5 * A + 0.5 * torch.eye(A.shape[1], device=A.device),
            rollout
        )
    return rollout
