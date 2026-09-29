"""Serve a GR00T-N1.6 checkpoint to the MESA / BiMESA evaluation server.

The MESA evaluation client (`scripts/eval_server_parallel.py` in the MESA repo) connects over a websocket using the
openpi-client protocol and sends observations of the form

    {"images": {<camera name>: (H, W, 3) uint8}, "state": (D,) float, "prompt": str}

and expects {"actions": (T, action_dim)} back. The camera names, state layout, and action keys are read from the
checkpoint's processor config / statistics, so the same server works for MESA (single-arm) and BiMESA (bimanual)
checkpoints.

Usage:
    uv run python gr00t/eval/serve_mesa.py --model-path <checkpoint dir> --port 8001
"""

import asyncio
import dataclasses
import http
import json
import logging
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import tyro
import websockets
import websockets.asyncio.server as _server
import websockets.frames

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval import msgpack_numpy
from gr00t.policy.gr00t_policy import Gr00tPolicy


@dataclasses.dataclass
class Args:
    model_path: str
    """Path to a GR00T-N1.6 checkpoint directory (containing config.json, processor_config.json, statistics.json)."""

    port: int = 8001
    host: str = "0.0.0.0"
    device: str = "cuda"

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag the checkpoint was finetuned with (MESA/BiMESA checkpoints use NEW_EMBODIMENT)."""

    exit_on_first_disconnect: bool = False


def _client_camera_name(modality_key: str) -> str:
    """Map a GR00T video key to the camera name used by the MESA client.

    e.g. "observation.images.egocentric" -> "egocentric", "leftshoulder_image" -> "leftshoulder".
    """
    name = modality_key.removeprefix("observation.images.")
    return name.removesuffix("_image")


class MESAGr00tPolicy:
    """Adapts MESA client observations to Gr00tPolicy inputs and back."""

    def __init__(self, args: Args):
        self._policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag, model_path=args.model_path, device=args.device
        )
        configs = self._policy.modality_configs
        self.video_keys = list(configs["video"].modality_keys)
        self.state_keys = list(configs["state"].modality_keys)
        self.action_keys = list(configs["action"].modality_keys)
        self.language_key = configs["language"].modality_keys[0]

        # The client sends one flat state vector; split it into the checkpoint's state keys in order.
        stats = json.loads((Path(args.model_path) / "statistics.json").read_text())
        state_stats = stats[args.embodiment_tag.value]["state"]
        self.state_dims = [len(state_stats[k]["mean"]) for k in self.state_keys]

        self.metadata = {
            "server": "gr00t_mesa",
            "cameras": [_client_camera_name(k) for k in self.video_keys],
            "state_keys": self.state_keys,
            "state_dim": int(sum(self.state_dims)),
            "action_keys": self.action_keys,
            "action_horizon": len(configs["action"].delta_indices),
        }
        logging.info("Policy metadata: %s", self.metadata)

    def infer(self, obs: dict) -> dict:
        video = {}
        for key in self.video_keys:
            image = np.asarray(obs["images"][_client_camera_name(key)])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Image for '{key}' must have shape (H, W, 3), got {image.shape}")
            video[key] = image.astype(np.uint8, copy=False)[None, None]

        flat_state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if flat_state.shape[0] != sum(self.state_dims):
            raise ValueError(f"Expected a {sum(self.state_dims)}-D state, got {flat_state.shape[0]}-D")
        state = {}
        for key, part in zip(self.state_keys, np.split(flat_state, np.cumsum(self.state_dims)[:-1])):
            state[key] = part[None, None]

        gr00t_obs = {
            "video": video,
            "state": state,
            "language": {self.language_key: [[obs["prompt"]]]},
        }
        action_dict, _ = self._policy.get_action(gr00t_obs)
        actions = np.concatenate([np.asarray(action_dict[k])[0] for k in self.action_keys], axis=-1)
        return {"actions": actions.astype(np.float32)}


class WebsocketPolicyServer:
    """Websocket server compatible with openpi_client.websocket_client_policy (adapted from openpi)."""

    def __init__(self, policy: MESAGr00tPolicy, *, host: str, port: int, exit_on_first_disconnect: bool):
        self._policy = policy
        self._host = host
        self._port = port
        self._exit_on_first_disconnect = exit_on_first_disconnect
        self._server = None

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            self._server = server
            logging.info("Serving on %s:%d", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logging.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._policy.metadata))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                start = time.monotonic()
                result = self._policy.infer(obs)
                result["server_timing"] = {"infer_ms": 1000 * (time.monotonic() - start)}
                await websocket.send(packer.pack(result))
            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                if self._exit_on_first_disconnect and self._server is not None:
                    self._server.close()
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main(args: Args) -> None:
    policy = MESAGr00tPolicy(args)
    WebsocketPolicyServer(
        policy, host=args.host, port=args.port, exit_on_first_disconnect=args.exit_on_first_disconnect
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
