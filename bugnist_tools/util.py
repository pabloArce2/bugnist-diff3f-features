import torch


def safe_name(name):
    """File-name friendly version of a label: letters, digits, '-' and '_' only."""
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)


def pick_device(name=None):
    """The requested torch device, or the first GPU if there is one."""
    return torch.device(name or ("cuda:0" if torch.cuda.is_available() else "cpu"))


def prompts_per_input(prompts, count, input_flag):
    """One prompt per input file; a single prompt is reused for every file."""
    if len(prompts) == 1:
        return prompts * count
    if len(prompts) != count:
        raise ValueError(f"--prompt must have either one value or the same number of values as {input_flag}.")
    return prompts
