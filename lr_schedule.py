from bisect import bisect_right

def adjust_lr(optimizer, lr, model_name, ep):
    if model_name == 'bfe':
        if ep < 50:
            lr = 1e-4 * (ep // 5 + 1)
        elif ep < 200:
            lr = 1e-3
        elif ep < 300:
            lr = 1e-4
        else:
            lr = 1e-5
    else:
        warmup_factor = 1
        ep = ep - 1

        if ep < 10:
            alpha = ep / 10
            warmup_factor = 0.01 * (1 - alpha) + alpha

        lr = lr * warmup_factor * 0.1 ** bisect_right([40, 70], ep)

    for p in optimizer.param_groups:
        p['lr'] = lr