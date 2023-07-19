from utils.test_utils.ddpm_test_util import DDPMTestLoop
from utils import dist_util
import matplotlib.pyplot as plt
from utils.galaxy_data_utils.transform_util import *

class VICDDPMTestLoop(DDPMTestLoop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uv_dense = np.load("./data/uv_dense.npy")
        self.uv_dense = torch.tensor(self.uv_dense)

    def sample(self, batch_kwargs):
        cond = {
            k: batch_kwargs[k].to(dist_util.dev()) for k in ["image_dir", "uv_coords", "vis_sparse"]
        }

        samples = []
        while len(samples) * self.batch_size < self.num_samples_per_mask:
            sample = self.diffusion.sample_loop(
                self.model,
                (self.batch_size, 2, self.image_size, self.image_size),
                cond,
                clip=False
            )

            samples.append(sample.cpu().detach().numpy())

        samples = np.concatenate(samples, axis=0)
        samples = samples[: self.num_samples_per_mask]
        return samples

