import functools

from utils import dist_util, logger
from utils.train_utils.base_train_util import TrainLoop

from utils.galaxy_data_utils.transform_util import *


class UNetTrainLoop(TrainLoop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_process(self, batch):
        ksapce_c, args_dict = batch
        image_dir = args_dict["image_dir"]
        image = args_dict["image"]
        return image_dir, image

    def forward_backward(self, batch):
        batch, label = self.batch_process(batch)
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro_input = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_label = label[i: i + self.microbatch].to(dist_util.dev())
            last_batch = (i + self.microbatch) >= batch.shape[0]
            micro_output = self.ddp_model(micro_input)

            compute_loss = functools.partial(
                th.nn.functional.mse_loss,
                micro_output,
                micro_label,
            )

            if last_batch or not self.use_ddp:
                loss = compute_loss()
            else:
                with self.ddp_model.no_sync():
                    loss = compute_loss()

            logger.log_kv("loss", loss)
            self.mp_trainer.backward(loss)

