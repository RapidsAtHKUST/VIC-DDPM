import matplotlib.pyplot as plt

from utils.script_util import save_args_dict, load_args_dict
from utils.test_utils.base_test_util import *
from utils.galaxy_data_utils.transform_util import *


MAX_NUM_SAVED_SAMPLES = 5

test_num = 0
class DDPMTestLoop(TestLoop):

    def __init__(self, diffusion, image_size, num_samples_per_mask=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion = diffusion
        self.image_size = image_size
        self.num_samples_per_mask = num_samples_per_mask


    def run_loop(self):
        super().run_loop()

    def forward_backward(self, data_item):
        global test_num
        img, batch_kwargs = data_item
        file_name = batch_kwargs["file_name"]
        slice_index = test_num
        test_num = test_num+1
        samples_path = os.path.join(self.output_dir, file_name, f"slice_{slice_index}")
        if os.path.exists(samples_path):
            if os.path.exists(os.path.join(samples_path, "slice_information.pkl")):
                logger.log(f"have sampled for {file_name} slice {slice_index}")
                return
        else:
            os.makedirs(samples_path, exist_ok=True)

        k_samples = self.sample(batch_kwargs)
        self.save_samples(k_samples, samples_path, batch_kwargs)
        logger.log(f"complete sampling for {file_name} slice {slice_index}")

    def sample(self, batch_kwargs):
        """
        The sample process is defined in children class.
        """
        pass

    def save_samples(self, k_samples, samples_path, batch_kwargs):

        image = batch_kwargs["image"][0][0]
        image_dir = batch_kwargs["image_dir"][0][0]


        for to_save_image, name in zip([image, image_dir], ["image", "image_dir"]):
            plt.imsave(
                fname=os.path.join(samples_path, f"{name}.png"),
                arr=to_save_image, cmap="hot"
            )

        # save some samples, less than 5
        # for i in range(min(MAX_NUM_SAVED_SAMPLES, len(samples))):
        #     sample = np.abs(samples[i][0])
        #     plt.imsave(
        #         fname=os.path.join(samples_path, f"sample_{i + 1}.png"),
        #         arr=sample, cmap="hot"
        #     )

        mean_sample = np.mean(k_samples, axis=0)
        np.save(os.path.join(samples_path, "asample_mean.npy"), mean_sample)
        plt.imsave(fname=os.path.join(samples_path, "asample_mean.png"), arr=mean_sample[0], cmap="hot")
        # save all information
        # np.savez(os.path.join(samples_path, f"all_samples"), samples0)  # arr is not magnitude images
        # saved_args = {
        #     "scale_coeff": batch_kwargs["scale_coeff"],
        #     "slice_index": batch_kwargs["slice_index"],
        #     "image": batch_kwargs["image"][0:1, ...],
        # }
        # save_args_dict(saved_args, os.path.join(samples_path, "slice_information.pkl"))


def extract_slice_index(slice_index):
    return int(slice_index.split("_")[-1])
