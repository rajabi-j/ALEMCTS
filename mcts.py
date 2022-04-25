from mctslib import MCTS
from ale_py import ALEInterface
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import os
import os.path
import random
import tempfile

class ALENode:
    __slots__ = ("state", "parent", "_evaluation", "action_id", "_is_terminal")
    def __init__(self, state, parent, score, action_id, is_terminal):
        self.state = state
        self.parent = parent
        self._evaluation = score
        self.action_id = action_id
        self._is_terminal = is_terminal


    @classmethod
    def setup_interface(cls, rom_path, frame_skip, random_seed=None):
        interface = ALEInterface()
        if random_seed is not None:
            interface.setInt("random_seed", random_seed)
        interface.setInt("frame_skip", frame_skip)
        interface.setFloat("repeat_action_probability", 0)
        interface.loadROM(rom_path)
        cls.interface = interface
        cls.ale_action_set = interface.getMinimalActionSet()
        cls.action_space_size = len(cls.ale_action_set)
        cls.action_set = [i for i in range(len(cls.ale_action_set))]


    @classmethod
    def root(cls):
        state = cls.interface.cloneState()
        parent = None
        score = 0
        action = 0 # attribute start of game to NOOP
        is_terminal = cls.interface.game_over()
        return cls(state, parent, score, action, is_terminal)

    @classmethod 
    def from_parent(cls, parent, action_id):
        parent.sync()
        inc_reward = cls.interface.act(cls.action_set[action_id])
        new_state = cls.interface.cloneState()
        is_terminal = cls.interface.game_over()

        return cls(new_state, parent, parent._evaluation + inc_reward, action_id, is_terminal)

    def sync(self):
        self.interface.restoreState(self.state)

    def apply_action(self, action_id):
        return ALENode.from_parent(self, action_id)

    def get_legal_actions(self):
        return self.action_set


    def is_terminal(self):
        return self._is_terminal

    def evaluation(self):
        return self._evaluation

    def get_history(self):
        history = []
        node = self
        while node.parent:
            history.append(node)
            node = node.parent
        
        return list(reversed(history))

    def make_video(self, video_path):
        history = self.get_history()

        with tempfile.TemporaryDirectory() as png_dir:
            for i, n in enumerate(history[:-1]):
                n.sync()
                self.interface.act(history[i+1].action_id)
                fname = f"frame_{i}.png"
                self.interface.saveScreenPNG(f"{os.path.join(png_dir, fname)}")

            os.system(f"ffmpeg -framerate 55 -start_number 0 -i {png_dir}/frame_%d.png -pix_fmt yuv420p {video_path}")

    def __hash__(self):
        """TODO: check how to implement this given we use state now"""
        return hash(self.ram.tobytes())
    
    def __eq__(self, other):
        return self.state == other.state

    def __repr__(self):
        return f"{self.__class__.__name__}<{self._evaluation=}, {self.action_id=}>"

if __name__ == "__main__":
    parser = ArgumentParser(description="Run MCTS on ALE")

    args = [
        ("rom_path", {"type": str}),
        ("--exploration_weight", {"type": float, "required": True}),
        ("--cpu_time", {"type": float, "required": True}),
        ("--rollout_depth", {"type": int, "required": True}),
        ("--frame_skip", {"type": int, "required": True}),
        ("--turn_limit", {"type": int, "required": True}),
        ("--video_path", {"type": str, "required": True}),
        ("--structure", {"type": str, "required": True}),
        ("--tiebreak", {"type": str, "default": "random", "choices": ["first", "random"]}),
        ("--random_seed", {"type": int, "default": None}),
        ("--no_progress_bar", {"default": False, "action": "store_true"}),
    ]


    for name, opts in args:
        parser.add_argument(name, **opts)
    
    args = parser.parse_args()

    ALENode.setup_interface(args.rom_path, args.frame_skip, args.random_seed)

    mcts = MCTS(ALENode.root(), structure=args.structure, iter_stop="cpu_time", action_space_size=ALENode.action_space_size, constant_action_space=True, randomize_ties=True if args.tiebreak=="random" else false)

    turns = range(args.turn_limit) if args.no_progress_bar else tqdm(range(args.turn_limit))
    for i in turns:
        node = mcts.move(rollout_depth=args.rollout_depth, cpu_time=args.cpu_time, exploration_weight=args.exploration_weight)
        if not args.no_progress_bar:
            turns.set_description(f"node.evaluation: {node.evaluation()}", refresh=True)
        if node.is_terminal():
            break
    node.make_video(args.video_path)

