# ILVR + ADM

This is the implementation of [ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2108.02938).

This repository is heavily based on [guided diffusion](https://github.com/openai/guided-diffusion).

# Download pre-trained models

Here are the unconditional models trained on FFHQ and AFHQ-dog:

 * 256x256 FFHQ: [64x64_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt)
 * 256x256 AFHQ-dog: [64x64_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt)


# Sampling from pre-trained models

To sample from these models, you can use the `classifier_sample.py`, `image_sample.py`, and `super_res_sample.py` scripts.
Here, we provide flags for sampling from all of these models.
We assume that you have downloaded the relevant model checkpoints into a folder called `models/`.

For these examples, we will generate 100 samples with batch size 4. Feel free to change these values.

```
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
```

## Classifier guidance

Note for these sampling runs that you can set `--classifier_scale 0` to sample from the base diffusion model.
You may also use the `image_sample.py` script instead of `classifier_sample.py` in that case.

 * 64x64 model:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt --model_path models/64x64_diffusion.pt $SAMPLE_FLAGS
```

 * 128x128 model:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 0.5 --classifier_path models/128x128_classifier.pt --model_path models/128x128_diffusion.pt $SAMPLE_FLAGS
```

 * 256x256 model:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS
```

 * 256x256 model (unconditional):

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS
```

 * 512x512 model:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 512 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 4.0 --classifier_path models/512x512_classifier.pt --model_path models/512x512_diffusion.pt $SAMPLE_FLAGS
```

## Upsampling

For these runs, we assume you have some base samples in a file `64_samples.npz` or `128_samples.npz` for the two respective models.

 * 64 -&gt; 256:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --large_size 256  --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python super_res_sample.py $MODEL_FLAGS --model_path models/64_256_upsampler.pt --base_samples 64_samples.npz $SAMPLE_FLAGS
```

 * 128 -&gt; 512:

```
MODEL_FLAGS="--attention_resolutions 32,16 --class_cond True --diffusion_steps 1000 --large_size 512 --small_size 128 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python super_res_sample.py $MODEL_FLAGS --model_path models/128_512_upsampler.pt $SAMPLE_FLAGS --base_samples 128_samples.npz
```

## LSUN models

These models are class-unconditional and correspond to a single LSUN class. Here, we show how to sample from `lsun_bedroom.pt`, but the other two LSUN checkpoints should work as well:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python image_sample.py $MODEL_FLAGS --model_path models/lsun_bedroom.pt $SAMPLE_FLAGS
```

You can sample from `lsun_horse_nodropout.pt` by changing the dropout flag:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python image_sample.py $MODEL_FLAGS --model_path models/lsun_horse_nodropout.pt $SAMPLE_FLAGS
```

Note that for these models, the best samples result from using 1000 timesteps:

```
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 1000"
```

# Results

These are samples generated with N=8 and 16:

![a](gif/full_face.gif)

These are cat-to-dog samples generated with N=32:

![b](gif/full_cat2dog.gif)


# Training models

Training diffusion models is described in the [parent repository](https://github.com/openai/improved-diffusion). Training a classifier is similar. We assume you have put training hyperparameters into a `TRAIN_FLAGS` variable, and classifier hyperparameters into a `CLASSIFIER_FLAGS` variable. Then you can run:

```
mpiexec -n N python scripts/classifier_train.py --data_dir path/to/imagenet $TRAIN_FLAGS $CLASSIFIER_FLAGS
```

Make sure to divide the batch size in `TRAIN_FLAGS` by the number of MPI processes you are using.

Here are flags for training the 128x128 classifier. You can modify these for training classifiers at other resolutions:

```sh
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 256 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
```

For sampling from a 128x128 classifier-guided model, 25 step DDIM:

```sh
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --image_size 128 --learn_sigma True --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True"
SAMPLE_FLAGS="--batch_size 4 --num_samples 50000 --timestep_respacing ddim25 --use_ddim True"
mpiexec -n N python scripts/classifier_sample.py \
    --model_path /path/to/model.pt \
    --classifier_path path/to/classifier.pt \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS
```

To sample for 250 timesteps without DDIM, replace `--timestep_respacing ddim25` to `--timestep_respacing 250`, and replace `--use_ddim True` with `--use_ddim False`.
