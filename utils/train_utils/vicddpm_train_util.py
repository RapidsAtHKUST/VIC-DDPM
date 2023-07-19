from utils.train_utils.ddpm_train_util import *


class VICDDPMTrainLoop(DDPMTrainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_process(self, batch):
        # modify condition so that it only contains the information we need.
        batch, cond = batch
        cond = {
            k: cond[k] for k in ["uv_coords", "image_dir", "vis_sparse"]
        }
        return batch, cond
