def adjust_learning_rate(optimizer, epoch, opt):

    lr_shceduler(optimizer, epoch, opt.lr)

def lr_shceduler(optimizer, epoch, init_lr):

    if epoch > 36:
        init_lr *= 0.5e-3
    elif epoch > 32:
        init_lr *= 1e-3
    elif epoch > 24:
        init_lr *= 1e-2
    elif epoch > 16:
        init_lr *= 1e-1
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr