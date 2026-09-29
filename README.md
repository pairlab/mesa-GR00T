# GR00T-N1.6 inference for MESA / BiMESA

This is a minimal, inference-only fork of [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (N1.6,
upstream commit `d331b68`) for serving the GR00T-N1.6 baselines from the MESA paper to the
[MESA](https://pairlab.github.io/MESA/) evaluation server. Training code, simulation harnesses, and examples from the
upstream repository have been removed; see upstream for training and finetuning.

Changes relative to upstream:
- `gr00t/eval/serve_mesa.py`: websocket policy server speaking the openpi-client protocol used by MESA's
  `scripts/eval_server_parallel.py`. Camera names, state layout, and action keys are read from the checkpoint.
- `gr00t/data/state_action/state_action_processor.py`: MESA/BiMESA relative joint actions keep the gripper dimensions
  absolute (`_mesa_joint_reference`). The released checkpoints were trained with this convention.
- `pyproject.toml` / `uv.lock`: `transformers==4.51.3` (the version used to train and evaluate the checkpoints) and
  `websockets`.

## Installation

Requires Python 3.10, CUDA 12, and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> gr00t-mesa && cd gr00t-mesa
uv sync
```

## Serving a checkpoint

MESA and BiMESA GR00T-N1.6 checkpoints are finetuned from `nvidia/GR00T-N1.6-3B` with the `new_embodiment` tag and
store their modality configuration and normalization statistics in the checkpoint directory. Download a checkpoint
(links on the [MESA documentation](https://pairlab.github.io/MESA/)) and run:

```bash
uv run python gr00t/eval/serve_mesa.py --model-path <checkpoint dir> --port 8001
```

Then, from the MESA repository, run the evaluation server against the same port.

Single-arm MESA (Franka; left-shoulder + wrist cameras; 8-D joint-position state/action):

```bash
uv run scripts/eval_server_parallel.py --port 8001 --eval-set-name mesa-70 \
  --num-rollouts-per-task 50 --controller-type joint_pos
```

Bimanual BiMESA (two ReverseMountedYam arms; egocentric + two wrist cameras; 14-D joint-position state/action):

```bash
uv run scripts/eval_server_parallel.py --port 8001 --eval-set-name bimesa-id \
  --num-rollouts-per-task 50 --controller-type joint_pos \
  --robots ReverseMountedYam ReverseMountedYam \
  --camera-names egocentric robot0_eye_in_hand robot1_eye_in_hand \
  --state-keys robot0_joint_pos robot0_gripper_jaw_width robot1_joint_pos robot1_gripper_jaw_width
```

The policy predicts 20-step action chunks at 20 Hz; the MESA evaluation server executes the first 5 actions before
replanning.

## License

Code is released under the upstream [Isaac-GR00T license](LICENSE). GR00T-N1.6 model weights are subject to NVIDIA's
model license; see [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B).
