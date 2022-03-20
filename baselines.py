# Random-action and NOOP baselines for ALE
# David Dunleavy, 2021-2022
# Mark Nelson, 2022

from ale_py import ALEInterface, Action
from argparse import ArgumentParser
from itertools import count
import numpy as np
import os
import tempfile

class BaselineAgent:
    def __init__(self, rom_path, baseline, turn_limit=None, frame_skip=None, video_path=None):
        self._ale = ALEInterface()
        self._turn_limit = turn_limit
        self._video_path = video_path
        self._baseline = baseline

        if frame_skip:
            self._ale.setInt("frame_skip", frame_skip)

        self._ale.setFloat("repeat_action_probability", 0)
        self._ale.loadROM(rom_path)

    def play(self):
        print("frame_count,score")
        score = 0
        min_action_set = self._ale.getMinimalActionSet()
        total_turns = range(self._turn_limit) if self._turn_limit else count()

        with tempfile.TemporaryDirectory() as png_dir:
            for i in total_turns:
                if self._ale.game_over():
                    break

                action = Action.NOOP if self._baseline=="noop" else np.random.choice(min_action_set)
                score += self._ale.act(action)
                self._ale.saveScreenPNG(os.path.join(png_dir, f"frame_{i}.png"))
                print(self._ale.getFrameNumber(), score)
                
            print("score:", score)
            os.system(f"ffmpeg -framerate 55 -start_number 0 -i {png_dir}/frame_%d.png -pix_fmt yuv420p {self._video_path}")



if __name__ == "__main__":
    parser = ArgumentParser(description="Run random or noop baseline on ALE")

    args = [
        ("rom_path", {"type": str}),
        ("baseline", {"type": str, "choices": ["noop", "random"]}),
        ("--frame_skip", {"type": int, "required": True}),
        ("--turn_limit", {"type": int}),
        ("--video_path", {"type": str, "required": True}),
    ]

    for name, opts in args:
        parser.add_argument(name, **opts)
    
    args = parser.parse_args()

    agent = BaselineAgent(args.rom_path, args.baseline, turn_limit=args.turn_limit, frame_skip=args.frame_skip, video_path=args.video_path)
    agent.play()
