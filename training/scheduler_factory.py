import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Optimizer

class SchedulerFactory:
    """
    Dynamically instantiates Learning Rate Schedulers.
    """

    @staticmethod
    def create(scheduler_name: str, optimizer: Optimizer, epochs: int, min_lr: float):
        name = scheduler_name.lower()
        
        if name == "cosineannealing":
            return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        elif name == "reducelronplateau":
            return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=min_lr)
        elif name == "onecyclelr":
            # Assuming steps_per_epoch is 1 for default instantiation (adjusted dynamically later)
            return lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'], 
                                           total_steps=epochs)
        elif name == "polynomialdecay":
            return lr_scheduler.PolynomialLR(optimizer, total_iters=epochs, power=1.0)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
