"""Record one CPU rollout as JSON for the SO-101 classroom website."""
import argparse
import dataclasses
import json
from pathlib import Path

import torch
from mujoco_playground._src.rl import env_registry
from mujoco_playground._src.rl.rsl_rl import checkpoint, vecenv_wrapper
from reaf.simulation.torch import flat_scene as config
from rsl_rl.runners import OnPolicyRunner


def record(args):
    entry = env_registry.get("SO101PickCube")
    runner_cfg = dataclasses.replace(entry.runner_cfg, nworld=1)
    cfg = entry.env_cfg_fn()
    cfg.sim.nworld = 1
    cfg.sim.xpu_device = "cpu"
    cfg.sim.impl = "warp"
    # A separate observation group leaves the checkpoint's actor and critic inputs intact.
    keys = ("ee/xpos", "object/xpos", "target/mocap_pos")
    cfg.observation_adapter["trace"] = config.ObsGroup(
        terms={key.split("/")[0]: config.Obs(key).configure(output_shape=(3,))
               for key in keys}, enable_noise=False)
    wrapped = vecenv_wrapper.RslRlVecEnvWrapper(config.build_env(cfg), cfg=runner_cfg.vecenv_wrapper_cfg)
    try:
        runner = OnPolicyRunner(wrapped, dataclasses.asdict(runner_cfg), device="cpu")
        runner.load(checkpoint.read_checkpoint(str(args.checkpoint)), map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")
        wrapped.seed(args.seed)
        obs, _ = wrapped.reset()
        initial = obs["trace"][0].flatten().tolist()
        frames = []
        dt = wrapped.unwrapped.device_layer.control_timestep
        with torch.inference_mode():
            for _ in range(wrapped.max_episode_length):
                # Step REAF directly so the final sample precedes the wrapper's auto-reset.
                action = vecenv_wrapper.split_actions(policy(obs), wrapped.action_spec())
                ts = wrapped.unwrapped.step(action)
                xyz = ts.observation["trace"][0].flatten().tolist()
                terms = {k: float(v.flatten()[0]) for k, v in ts.reward["terms"].items()
                         if not k.startswith("weighted/")}
                frames.append(dict(ee=xyz[:3], cube=xyz[3:6], goal=xyz[6:9], terms=terms,
                                   reward=float(ts.reward["reward"].flatten()[0])))
                if bool(ts.last().item()):
                    break
                obs = type(obs)(ts.observation, batch_size=[1])
        return dict(schema="so101.trace.v1", checkpoint=args.checkpoint.name, seed=args.seed,
                    initial=initial, dt=float(dt),
                    weights={k: float(v.weight) for k,v in cfg.task_layer.rewards.items()},
                    frames=frames)
    finally:
        wrapped.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = record(args)
    with args.output.open("x") as out:
        json.dump(result, out, allow_nan=False)
    print(f"Saved {len(result['frames'])} steps to {args.output}")
