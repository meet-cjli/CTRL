from .SimCLR.simclr import SimCLRModel
from .BYOL.byol import BYOL

def set_model(args):
    if args.method == 'simclr':
        return SimCLRModel(args)
    elif args.method == 'byol':
        return BYOL(args)
    else:
        raise  NotImplementedError

