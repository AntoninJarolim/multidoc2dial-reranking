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


def att_cat(outputs, out_grads, atts):
    """ Computes AttCAT scores (Attentive CAT)
    https://proceedings.neurips.cc/paper_files/paper/2022/file/20e45668fefa793bd9f2edf19be12c4b-Paper-Conference.pdf
    Warning: Most likely does not work for Batch > 1.
    :param outputs:  (L, B, S, E) - Layers, Batch, Sequence Length, Embedding size
    :param out_grads: (L, B, S, E) - Layers, Batch, Sequence Length, Embedding size
    :param atts: attentions: (L, H, S, S) - Layers, Heads, Sequence Length, Sequence Length
    :return: (S) AttCAT scores for each token
    """

    att_cats = []
    for layer in range(len(out_grads)):
        att = atts[layer]
        att = att.mean(dim=0)
        att = att.mean(dim=0)

        cat = (outputs[layer] * out_grads[layer]).sum(dim=-1).squeeze(0)
        att_cat = att * cat

        att_cats.append(att_cat)

    att_cats = torch.stack(att_cats)

    # Faster implementation of att cat
    all_attts = atts.mean(dim=(1, 2))  # Mean over head and first SL dims
    out_grads = torch.stack(out_grads)
    outputs = torch.stack(outputs)
    cat = (outputs * out_grads).sum(dim=-1).squeeze(1)
    att_cats2 = all_attts * cat

    # assert both methods compute the same
    assert torch.allclose(att_cats, att_cats2)

    att_cat_sum_layers = att_cats2.sum(dim=0)

    return att_cat_sum_layers
