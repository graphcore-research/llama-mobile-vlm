# Llama-Mobile

Code and release artifacts for [Llama-Mobile: Efficient 2.7-Bit Quantization of VLMs](https://arxiv.org/abs/2608.21134).

Including:

- C++ inference library for the demo app and benchmarking harness.
- Prebuilt model files for the inference library.
- Android demo app.
- Reference Quantization Aware Training (QAT) codebase.


## Repository Layout

- [`demo.py`](demo.py) - standalone HuggingFace demo for the QAT vision model.
- [`docs/`](docs) - specifies the `.sqt` file format and prebuilt model details.
- [`inference/`](inference) - supported C++ inference library, CLI, tests, and benchmarks.
- [`training/`](training) - reference quantization, training, evaluation, and model
  conversion code. This is included for transparency and provenance, not as a
  supported external training pipeline.
- [`android/`](android) - Android demo app source.


## Standalone Python Demo

For a standalone Python demo of the QAT vision model using Transformers, see [`demo.py`](demo.py). It loads Meta Llama 3.2 Vision Instruct, optionally applies the released packed S3D8 checkpoint, and can simulate INT8 activations.

```sh
pip install torch transformers accelerate safetensors pillow requests
mkdir -p models/hf
wget -P models/hf https://graphcore-research-public.s3.eu-west-1.amazonaws.com/2026-llama-mobile/hf_models/vision-11B-s3d8-packed.safetensors
echo "What is this an image of?" | python demo.py --checkpoint models/hf/vision-11B-s3d8-packed.safetensors --int8-activations --image https://picsum.photos/id/36/4179/2790
```

To run the original bfloat16 baseline model, omit `--checkpoint` and `--int8-activations`. Run without `echo ... |` to enter an interactive prompt. Note that the demo is not optimised for performance.


## Inference Library Quick Start

Install the C++ dependencies, then build and test the inference library:

```sh
sudo apt install clang clang-format gdb libomp-dev ninja-build
cd inference
./dev setup
./dev tests
```

Download prebuilt models:

```sh
mkdir -p models/
aws s3 cp --no-sign-request --region=eu-west-1 s3://graphcore-research-public/2026-llama-mobile/models/20260611/text-1B-int8.sqt models/
aws s3 cp --no-sign-request --region=eu-west-1 s3://graphcore-research-public/2026-llama-mobile/models/20260611/vision-11B-s3d8.sqt models/
# OR
wget https://graphcore-research-public.s3.eu-west-1.amazonaws.com/2026-llama-mobile/models/20260611/text-1B-int8.sqt -O models/text-1B-int8.sqt
wget https://graphcore-research-public.s3.eu-west-1.amazonaws.com/2026-llama-mobile/models/20260611/vision-11B-s3d8.sqt -O models/vision-11B-s3d8.sqt
```

Run text generation:

```sh
echo "What is blue?" | ./dev run cli -- models/text-1B-int8.sqt -g 128
```

Run vision-language generation based on TEST_IMAGE.jpg (replace with your own image):

```sh
echo "Describe this image." | ./dev run cli -- models/vision-11B-s3d8.sqt -g 64 --image TEST_IMAGE.jpg
```

Run a model-shaped benchmark:

```sh
./dev run benchmark -- text_model_1B
```

_Note that the inference library is optimised for Arm server and mobile CPUs - while it may run on x86, performance will be poor._

See [`inference/README.md`](inference/README.md) for more commands,
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for the model card, and
[`docs/models.md`](docs/models.md) for detailed conversion and download notes.


## Android App

The Android app in [`android/`](android) implements a runnable client demo, wrapping the C++ inference library in an Android interface.


## Training Code

The code in [`training/`](training) documents the quantization-aware training, evaluation, and `.sqt` conversion workflow used during the project. It depends on internal datasets, checkpoints, and infrastructure, so it is not presented as a supported external training setup.


## License

Copyright (c) 2026 Graphcore Ltd.

Built with Llama. Llama-Mobile is not affiliated with, sponsored by, or
endorsed by Meta.

The source code is licensed under the [MIT License](LICENSE). The derivative
model weights are licensed under the [Llama 3.2 Community License](LICENSE_LLAMA_3.2)
and include Meta's required [attribution notice](NOTICE). Use of Llama 3.2 is
also subject to Meta's [Acceptable Use Policy](https://www.llama.com/llama3_2/use-policy).
