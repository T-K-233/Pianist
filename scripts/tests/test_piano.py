# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Piano-Shadow-v0", help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import mido
import carb
import omni
import omni.appwindow
from isaaclab_tasks.utils import parse_env_cfg

import pianist.tasks  # noqa: F401
from pianist.tasks.manipulation.piano.mdp.commands import NUM_KEYS, WHITE_KEY_INDICES

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    midi_port = "Piano MIDI Device:Piano MIDI Device Digital Piano 24:0"

    inport = mido.open_input(midi_port)
    outport = mido.open_output(midi_port)

    query_key_names = []
    for i in range(NUM_KEYS):
        if i in WHITE_KEY_INDICES:
            query_key_names.append(f"white_key_{i}")
        else:
            query_key_names.append(f"black_key_{i}")

    # key_body_indices, _ = env.unwrapped.scene["piano"].find_bodies(query_key_names, preserve_order=True)
    key_joint_indices, _ = env.unwrapped.scene["piano"].find_joints([name + "_joint" for name in query_key_names], preserve_order=True)

    joint_targets = torch.zeros(1, env.unwrapped.scene["piano"].num_joints, device=env.unwrapped.device)
    keyboard_control_targets = torch.zeros_like(joint_targets)

    def _on_keyboard_event(event: carb.input.KeyboardEvent):
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed."""
        # print(event.type, event.input.name)
        if event.type not in [carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_RELEASE]:
            return

        match event.input.name:
            case "KEY_1":
                note_index = 60
            case "KEY_2":
                note_index = 62
            case "KEY_3":
                note_index = 64
            case "KEY_4":
                note_index = 65
            case "KEY_5":
                note_index = 67
            case "KEY_6":
                note_index = 69
            case "KEY_7":
                note_index = 71
            case "KEY_8":
                note_index = 72
            case "KEY_9":
                note_index = 74
            case "KEY_0":
                note_index = 76
            case "KEY_MINUS":
                note_index = 77
            case "KEY_EQUALS":
                note_index = 79
            case "KEY_BACKSPACE":
                note_index = 81
            case _:
                return

        key_index = note_index - 21
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            keyboard_control_targets[:, key_joint_indices[key_index]] = 1.0
            outport.send(mido.Message("note_on", note=note_index, velocity=127, channel=0))
            print("Send: note_on", note_index)
        else:
            keyboard_control_targets[:, key_joint_indices[key_index]] = 0.0
            outport.send(mido.Message("note_off", note=note_index, velocity=127, channel=0))
            print("Send: note_off", note_index)

    carb_input = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    carb_input.subscribe_to_keyboard_events(keyboard, _on_keyboard_event)

    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

            for msg in inport.iter_pending():
                if msg.channel != 0:
                    continue

                key_index = msg.note - 21
                if msg.type == "note_on":
                    print(f"Received: Note on: {msg.note}, Velocity: {msg.velocity}")
                    joint_targets[:, key_joint_indices[key_index]] = 1.0
                elif msg.type == "note_off":
                    print(f"Received: Note off: {msg.note}, Velocity: {msg.velocity}")
                    joint_targets[:, key_joint_indices[key_index]] = 0.0

            env.unwrapped.scene["piano"].set_joint_position_target(torch.max(joint_targets, keyboard_control_targets))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
