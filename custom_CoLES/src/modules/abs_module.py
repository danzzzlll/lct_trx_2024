import torch
from ptls.frames.abs_module import ABSModule


class ABSModuleCustom(ABSModule):
    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self.parameters())
        opt_dict = {'optimizer': optimizer}

        if self._lr_scheduler_partial is not None:
            scheduler = self._lr_scheduler_partial(optimizer)
        
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler = {
                    'scheduler': scheduler,
                    'monitor': self.metric_name,
                }

            opt_dict['lr_scheduler'] = scheduler

        return opt_dict