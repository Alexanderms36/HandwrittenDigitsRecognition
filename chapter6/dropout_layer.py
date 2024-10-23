from aesara.tensor.random.utils import RandomStream


def drp_lr(inpt_dropout, mini_batch_size, p_dropout, n_in):
    srng = RandomStream(seed=42)
    reshaped_input = inpt_dropout.reshape((mini_batch_size, n_in))
    dropout_mask = srng.binomial(n=1, size=reshaped_input.shape, p=1-p_dropout)
    return reshaped_input * dropout_mask / (1 - p_dropout)