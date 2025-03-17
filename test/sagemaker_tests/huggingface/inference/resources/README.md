# Commands to generate neuronx artifacts

## tiny-distilbert

``` shell
optimum-cli export neuron --model hf-internal-testing/tiny-random-DistilBertModel \
                          --batch_size 1 \
                          --sequence_length 128 \
                          --auto_cast_type bf16 \
                          tiny-distilbert-sst-2
```

## tiny-gpt2

```shell
optimum-cli export neuron -m hf-internal-testing/tiny-random-gpt2 \
                          --task text-generation \
                          --batch_size 1 \
                          --auto_cast_type bf16 \
                          --sequence_length 128 \
                          --num_cores 2 \
                          tiny-gpt2
```

## tiny-sdxl

```shell
optimum-cli export neuron -m echarlaix/tiny-random-stable-diffusion-xl \
                          --task stable-diffusion \
                          --batch_size 1 \
                          --auto_cast matmul \
                          --auto_cast_type bf16 \
                          --width 64 \
                          --height 64 \
                          --num_images_per_prompt 1 \
                          tiny-sdxl
```
