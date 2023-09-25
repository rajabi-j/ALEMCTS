from mctslib import MCTS
from ale_py import ALEInterface
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import os
import os.path
import random
import tempfile
import csv
import sys

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
    
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def save_to_csv(data, csv_file_path):
    """
    Save data to a CSV file.

    Args:
        data (list of lists): The data to be saved to the CSV file.
        csv_file_path (str): The path to the CSV file where data will be saved.
    """
    try:
        # Open the CSV file in write mode and specify the CSV writer
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write each row from the data list to the CSV file
            writer.writerow(data)

        print(f'Data has been saved to CSV file "{csv_file_path}" successfully.')
    except Exception as e:
        print(f'Error: {e}')


def mcts_run(args):

    print("Running test with the following parameters:")
    print(f"\tCpu_time: {args.cpu_time}")
    print(f"\tRollout_depth: {args.rollout_depth}")
    print(f"\tFrame_skip: {args.frame_skip}")
    print(f"\tTurn_limit: {args.turn_limit} \n")

    ALENode.setup_interface(args.rom_path, args.frame_skip, args.random_seed)

    mcts = MCTS(ALENode.root(), structure=args.structure, max_action_value=ALENode.action_space_size-1, constant_action_space=True, randomize_ties=True if args.tiebreak=="random" else False)

    turns = range(args.turn_limit) if args.no_progress_bar else tqdm(range(args.turn_limit))
    for i in turns:
        node, _, _ = mcts.search_using_cpu_time(rollout_depth=args.rollout_depth, cpu_time=args.cpu_time, exploration_weight=args.exploration_weight)
        mcts.choose_best_node()

        if not args.no_progress_bar:
            turns.set_description(f"node.evaluation: {node.state.evaluation()}", refresh=True)
        if node.state.is_terminal():
            break

    node.state.make_video(args.video_path)

    return node.state.evaluation()

    

if __name__ == "__main__":

    test_time = 0.1
    test_limit = 300
    test_skip = 5
    test_seed = 20230921

    min_depth = 100
    max_depth = 300
    depth_step = 100

    for test_depth in range(min_depth, max_depth, depth_step):

        test_path = video_path = 'videos/' + 'boxing' + '_depth' + str(test_depth).zfill(3) + '_limit' + str(test_limit).zfill(3) \
          + '_time' + str(test_time) + '_skip' + str(test_skip) + '.mp4'
        
        rom_name = 'boxing'

        args = Namespace(
          rom_path = 'roms/' + rom_name + '.bin',
          exploration_weight = 1.0,
          cpu_time = test_time,
          rollout_depth = test_depth,
          frame_skip = test_skip,
          turn_limit = test_limit,
          video_path = test_path,
          structure = 'tree',
          tiebreak = 'random',
          random_seed = test_seed,
          no_progress_bar = False
        )

        test_score = mcts_run(args)

        test_specs = [rom_name, args.rollout_depth, args.turn_limit, args.cpu_time, args.frame_skip, test_score, args.video_path]

        save_to_csv(test_specs, 'result.csv')

