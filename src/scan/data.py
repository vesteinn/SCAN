import torch
import torch.nn.functional as F


def parse_action_command(line):
    command, action = line.strip().split(" OUT: ")
    command = command[3:]  # Remove "IN: "
    return command, action


def generate_scan_dictionary(full_dataset, add_bos=True, add_eos=True):
    #
    # Example (split on separator):
    # IN: walk opposite right thrice after run opposite right
    # OUT: I_TURN_RIGHT I_TURN_RIGHT I_RUN I_TURN_RIGHT I_TURN_RIGHT I_WALK I_TURN_RIGHT I_TURN_RIGHT I_WALK I_TURN_RIGHT I_TURN_RIGHT I_WALK
    #
    command_dict = dict()
    action_dict = dict()

    commands = set()
    actions = set()

    with open(full_dataset) as infile:
        for line in infile.readlines():
            command, action = parse_action_command(line)
            commands.update(set(command.split()))
            actions.update(set(action.split()))

    # Sorted needed for deterministic id's
    for idx, action in enumerate(sorted(actions)):
        action_dict[action] = idx
    for idx, command in enumerate(sorted(commands)):
        command_dict[command] = idx

    action_dict["PAD"] = len(action_dict)
    command_dict["PAD"] = len(command_dict)

    if add_bos:
        action_dict["BOS"] = len(action_dict)
        command_dict["BOS"] = len(command_dict)
    if add_eos:
        action_dict["EOS"] = len(action_dict)
        command_dict["EOS"] = len(command_dict)

    return command_dict, action_dict


class SCANDataset(torch.utils.data.Dataset):
    def __init__(self, data, src_dict, tgt_dict, device):
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.device = device
        self.commands = []
        self.actions = []
        self.load(data)

    def __len__(self):
        return len(self.commands)

    def __getitem__(self, idx):
        return (self.commands[idx], self.actions[idx])

    def encode(self, text, dictionary, add_bos=False, add_eos=False):
        # TODO: read from variable
        if add_eos:
            text += " EOS"
        if add_bos:
            text = "BOS " + text
        encoding = [dictionary[a] for a in text.strip().split()]
        return torch.tensor(encoding).to(self.device)

    def encode_command(self, command):
        return self.encode(command, self.src_dict, add_eos=True)

    def encode_action(self, action):
        return self.encode(action, self.tgt_dict, add_eos=True)

    def load(self, data):
        with open(data) as infile:
            for line in infile.readlines():
                command, action = parse_action_command(line)
                self.commands.append(self.encode_command(command))
                self.actions.append(self.encode_action(action))


if __name__ == "__main__":
    # Simple test for ScanDataset, assumes submodule with data
    # has been loaded.
    tasks = "../../data/SCAN/tasks.txt"
    src_dict, tgt_dict = generate_scan_dictionary(tasks)
    assert len(src_dict) > 0
    assert len(tgt_dict) > 0
    print(
        f"Dictionaries lodaded, {len(src_dict)} in src_dict, {len(tgt_dict)} in tgt_dict."
    )

    dataset = SCANDataset(tasks, src_dict, tgt_dict, "cpu")
    assert len(dataset) > 2000
    print(f"Dataset loaded with {len(dataset)} records.")
