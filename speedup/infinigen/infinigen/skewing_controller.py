import torch


def weight_bias_concat(weight, bias, scaling=False, head_dim=1.0):
    """Concatenates the weight matrix and bias.

    On the warmup phase, concatenates the weight matrix and bias for skewing.
    This manipulation does not hurt the correctness.

    Args:
        weight: Weight matrix (D, D)
        bias: Bias vector (D)
        scaling: If ture, scales the concatenated weight and bias to skip
            the scaling after projection.
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        concatenated weight and bias (D, D+1)
    """

    if scaling:
        return torch.cat((weight, bias.unsqueeze(1).to(weight.device)), dim=1) * (
            head_dim**-0.5
        )
    else:
        return torch.cat((weight, bias.unsqueeze(1).to(weight.device)), dim=1)


def reform_hidden_states(hidden_states):
    """Concatenates the weight matrix and bias.

    Concatenates the hidden states with a column of 1.
    This reformation with the concatenated weight and bias  makes the linear
    projection into a one matrix multiplication without bias addition.

    Args:
        hidden: Hidden states (b, n, D)

    Returns:
        reformed hidden states (b, n, D+1)
    """

    return torch.cat(
        (hidden_states, torch.ones_like(hidden_states)[:, :, 1].unsqueeze(2)), dim=-1
    )


def skew(query, key, wq, wk, n_head, head_dim):
    """Manipulates the query/key weight matrix for skewing the qeury and key matrix.

    On the warmup phase, manipulates the query/key weight matrix for
    skewing the query and key matrix. By doing so, a few columns of
    the query and key matrix have become much more important. We use
    the columns for attention speculation.

    Args:
        query: Query matrix (b, n, h, d)
        key: Key matrix (b, n, h, d)
        w_q: Concatenated query weight and bias (D, D+1)
        w_k: Concatenated key weight and bias (D, D+1)
        n_head: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        w_q: Manipulated w_q (D, D+1)
        w_k: Manipulated w_k (D, D+1)

    """

    for h_idx in range(n_head):
        start = h_idx * head_dim
        end = (h_idx + 1) * head_dim
        _, sq, vq = torch.svd(query[0, :, h_idx].to(torch.float))
        _, sk, _ = torch.svd(key[0, :, h_idx].to(torch.float))
        sq = sq.to(torch.float16)
        vq = vq.to(torch.float16)
        sk = sk.to(torch.float16)
        sq = sq * sk
        A = torch.zeros(head_dim, head_dim).to(query.device).to(torch.float16)
        _, ind = sq.sort()
        A = A.scatter(-1, ind.unsqueeze(0).repeat(head_dim, 1), vq)
        wq[start:end, :] = A.t() @ wq[start:end]
        wk[start:end, :] = A.t() @ wk[start:end]
    return wq, wk
