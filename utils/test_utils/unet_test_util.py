import matplotlib.pyplot as plt

from utils.test_utils.base_test_util import *
from utils.galaxy_data_utils.transform_util import *

class UNetTestLoop(TestLoop):

    def __init__(self, microbatch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.microbatch = microbatch
        assert microbatch >= 1
        assert self.batch_size == 1

        self.curr_file = {
            "file_name": "",
            "image": [],
            "image_dir": [],
            "vis_zf": [],
            "mask": [],
            "scale_coeff": [],
            "slice_index": [],
        }

    def run_loop(self):
        super().run_loop()
        self.reconstruct_for_one_volume()


    def reconstruct_for_one_volume(self):
        for key in ["image", "image_dir", "vis_zf", "mask"]:
            self.curr_file[key] = th.cat(self.curr_file[key], dim=0).to(dist_util.dev())
        outputs = []
        for i in range(0, len(self.curr_file["slice_index"]), self.microbatch):
            micro_input = self.curr_file["image_dir"][i: i + self.microbatch]
            with th.no_grad():
                micro_output = self.model(micro_input)
            outputs.append(micro_output)
        outputs = th.cat(outputs, dim=0)
        outputs_refine = outputs

        for i in range(len(self.curr_file["slice_index"])):
            # rescale image value
            scale_coeff = self.curr_file["scale_coeff"][i]
            self.curr_file["image"][i] = self.curr_file["image"][i] / scale_coeff
            outputs[i] = outputs[i] / scale_coeff
            outputs_refine[i] = outputs_refine[i] / scale_coeff

            # make directories
            dir_path = os.path.join(
                self.output_dir, self.curr_file["file_name"],
                f"slice_{self.curr_file['slice_index'][i]}"
            )
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            # compute metrics for slices
            # outputs
            curr_slice_metrics = compute_metrics(self.curr_file["image"][i: i + 1], outputs[i: i + 1])
            for key in METRICS:
                self.slice_metrics[key].append(curr_slice_metrics[key])
            write_metric_to_file(
                os.path.join(dir_path, "output_metrics.txt"),
                curr_slice_metrics,
                f"volume {self.curr_file['file_name']}, slice {self.curr_file['slice_index'][i]}, unet output\n"
            )

            # outputs_refine
            curr_slice_metrics = compute_metrics(self.curr_file["image"][i: i + 1], outputs_refine[i: i + 1])
            for key in METRICS:
                self.refined_slice_metrics[key].append(curr_slice_metrics[key])
            write_metric_to_file(
                os.path.join(dir_path, "refined_output_metrics.txt"),
                curr_slice_metrics,
                f"volume {self.curr_file['file_name']}, slice {self.curr_file['slice_index'][i]}, unet refined output\n"
            )

        # compute metrics for volume
        # outputs
        curr_volume_metrics = compute_metrics(self.curr_file["image"], outputs)
        for key in METRICS:
            self.volume_metrics[key].append(curr_volume_metrics[key])
        write_metric_to_file(
            os.path.join(self.output_dir, self.curr_file["file_name"], "output_metrics.txt"),
            curr_volume_metrics,
            f"volume {self.curr_file['file_name']}, unet output\n"
        )

        # outputs_refine
        curr_volume_metrics = compute_metrics(self.curr_file["image"], outputs_refine)
        for key in METRICS:
            self.refined_volume_metrics[key].append(curr_volume_metrics[key])
        write_metric_to_file(
            os.path.join(self.output_dir, self.curr_file["file_name"], "refined_output_metrics.txt"),
            curr_volume_metrics,
            f"volume {self.curr_file['file_name']}, unet refined output\n"
        )

        # save outputs
        self.save_outputs(outputs, outputs_refine)

    def forward_backward(self, data_item):
        ksapce_c, batch_args = data_item

        # reconstruct for the whole volume
        if self.curr_file["file_name"] != batch_args["file_name"] and self.curr_file["file_name"] != "":
            self.reconstruct_for_one_volume()

        if self.curr_file["file_name"] != batch_args["file_name"]:
            for key in self.curr_file.keys():
                self.curr_file[key] = []
            self.curr_file["file_name"] = batch_args["file_name"]

        for key in self.curr_file.keys():
            if key == "file_name":
                continue
            else:
                self.curr_file[key].append(batch_args[key])

    def save_outputs(self, outputs, outputs_refine):
        assert self.curr_file
        for i in range(len(self.curr_file["slice_index"])):
            dir_path = os.path.join(
                self.output_dir, self.curr_file["file_name"],
                f"slice_{self.curr_file['slice_index'][i]}"
            )
            mask = th2np(self.curr_file["mask"][i, 0, ...])
            image = th2np_magnitude(self.curr_file["image"][i:i + 1])[0]
            image_dir = th2np_magnitude(self.curr_file["image_dir"][i:i + 1])[0]
            output = th2np_magnitude(outputs[i:i + 1])[0]
            output_refine = th2np_magnitude(outputs_refine[i:i + 1])[0]

            to_save_images = [mask, image, image_dir, output, output_refine]
            to_save_images_name = ["mask", "image", "image_dir", "output", "output_refine"]
            for to_save_image, name in zip(to_save_images, to_save_images_name):
                plt.imsave(
                    fname=os.path.join(dir_path, f"{name}.png"),
                    arr=to_save_image, cmap="gray"
                )
