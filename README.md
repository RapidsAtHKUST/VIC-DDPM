# README

Code for 'A Conditional Denoising Diffusion Probabilistic Model for Radio Interferometic Image Reconstruction'.
Preprint: https://arxiv.org/abs/2305.09121 .

### Training

```
SCRIPT_FLAGS="--method_type vicddpm"
DATASET_FLAGS="--dataset galaxy --batch_size 24 --num_workers 6"
TRAIN_FLAGS="--microbatch 32 --save_interval 5000 --max_step 25000 \
--model_save_dir ..."

python -m torch.distributed.launch --nproc_per_node=6 train.py $SCRIPT_FLAGS $DATASET_FLAGS $TRAIN_FLAGS
```

### Testing

```
SCRIPT_FLAGS="--method_type vicddpm"
DATASET_FLAGS="--dataset galaxy \
--batch_size 1 --num_workers 2"
TEST_FLAGS="--model_save_dir ... --resume_checkpoint model025000.pt \
--output_dir ... \
--debug_mode False"

python -m torch.distributed.launch --nproc_per_node=6 test.py $SCRIPT_FLAGS $DATASET_FLAGS $TEST_FLAGS
```

### Dataset

We use public dataset which is presented by Wu et al.[1]. Please find the dataset in https://github.com/wubenjamin/neural-interferometry .

### Reference

[1] Wu, Benjamin, et al. "Neural Interferometry: Image Reconstruction from Astronomical Interferometers using Transformer-Conditioned Neural Fields." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 36. No. 3. 2022.
