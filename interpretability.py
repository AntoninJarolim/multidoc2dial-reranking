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


def grad_sam(attentions, gradients):
    """ Computes Grad-SAM (Gradient-weighted Self-Attention Map)
    https://arxiv.org/pdf/2204.11073
    :param attentions: (L, H, S, S) - Layers, Heads, Sequence Length, Sequence Length
    :param gradients: (L, H, S, S) - Layers, Heads, Sequence Length, Sequence Length
    :return: (S) Grad-SAM scores for each token
    """

    # Compute the gradient-weighted attention matrix
    relu_fce = torch.nn.ReLU()
    weighted_attentions = attentions * relu_fce(gradients)
    return torch.mean(weighted_attentions, dim=(0, 1, 2))
