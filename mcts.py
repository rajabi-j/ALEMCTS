from mctslib import MCTS
from ale_py import ALEInterface
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import os
import os.path
import random

class ALENode:
    __slots__ = ("state", "parent", "_evaluation", "action", "_is_terminal")
    def __init__(self, state, parent, score, action, is_terminal):
        self.state = state
        self.parent = parent
        self._evaluation = score
        self.action = action
        self._is_terminal = is_terminal


    @classmethod
    def setup_interface(cls, rom_path, frame_skip):
        interface = ALEInterface()
        interface.setInt("frame_skip", frame_skip)
        interface.setFloat("repeat_action_probability", 0)
        interface.loadROM(rom_path)
        cls.interface = interface

    @classmethod
    def root(cls):
        state = cls.interface.cloneState()
        parent = None
        score = 0
        action = 0 # attribute start of game to NOOP
        is_terminal = cls.interface.game_over()
        return cls(state, parent, score, action, is_terminal)

    @classmethod 
    def from_parent(cls, parent, action):
        parent.sync()
        inc_reward = cls.interface.act(action)
        new_state = cls.interface.cloneState()
        is_terminal = cls.interface.game_over()

        return cls(new_state, parent, parent._evaluation + inc_reward, action, is_terminal)

    def sync(self):
        self.interface.restoreState(self.state)
    
    def find_children(self):
        actions = self.interface.getMinimalActionSet()

        return [ALENode.from_parent(self, int(a)) for a in actions]

    def is_terminal(self):
        return self._is_terminal

    def evaluation(self):
        return self._evaluation

    def random_child(self):
        action = random.choice(self.interface.getMinimalActionSet())
        return ALENode.from_parent(self, int(action))

    def get_history(self):
        history = []
        node = self
        while node.parent:
            history.append(node)
            node = node.parent
        
        return list(reversed(history))

    def make_video(self, png_dir, video_path):
        history = self.get_history()

        for i, n in enumerate(history[:-1]):
            n.sync()
            self.interface.act(history[i+1].action)
            fname = f"frame_{i}.png"
            self.interface.saveScreenPNG(f"{os.path.join(png_dir, fname)}")

        os.system(f"ffmpeg -framerate 30 -start_number 0 -i {png_dir}/frame_%d.png -pix_fmt yuv420p {video_path}")
        os.system(f"rm -rf {png_dir}/*")

    def __hash__(self):
        """TODO: check how to implement this given we use state now"""
        return hash(self.ram.tobytes())
    
    def __eq__(self, other):
        return self.state == other.state

    def __repr__(self):
        return f"{self.__class__.__name__}<{self._evaluation=}, {self.action=}>"

if __name__ == "__main__":
    parser = ArgumentParser(description="Run MCTS on ALE")

    args = [
        ("rom_path", {"type": str}),
        ("--exploration_weight", {"type": float, "required": True}),
        ("--cpu_time", {"type": float, "required": True}),
        ("--rollout_depth", {"type": int, "required": True}),
        ("--frame_skip", {"type": int, "required": True}),
        ("--turn_limit", {"type": int, "required": True}),
        ("--png_dir", {"type": str, "required": True}),
        ("--video_path", {"type": str, "required": True}),
    ]


    for name, opts in args:
        parser.add_argument(name, **opts)
    
    args = parser.parse_args()

    ALENode.setup_interface(args.rom_path, args.frame_skip)

    mcts = MCTS(ALENode.root(), structure="tree", iter_stop="cpu_time")

    for i in (bar := tqdm(range(args.turn_limit))):
        node = mcts.move(rollout_depth=args.rollout_depth, cpu_time=args.cpu_time)
        bar.set_description(f"node.evaluation: {node.evaluation()}", refresh=True)
    node.make_video(args.png_dir, args.video_path)

