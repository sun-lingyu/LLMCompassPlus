import base64
import os
import shlex
import subprocess

file_dir = os.path.dirname(os.path.abspath(__file__))


test_model_dict = {
    "InternVision": {
        "head_dim": 64,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
    },
    "Qwen3_0_6B": {
        "head_dim": 128,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "hidden_act": "silu",
    },
    "Qwen3_1_7B": {
        "head_dim": 128,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "hidden_act": "silu",
    },
    "Qwen3_4B": {
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "hidden_act": "silu",
    },
    "Qwen3_8B": {
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "hidden_act": "silu",
    },
}


def run_remote_command(user, host, port, remote_cmd, work_dir=None):
    remote_cmd_str = " ".join(shlex.quote(arg) for arg in remote_cmd)
    if work_dir is not None:
        remote_cmd_str = f"cd {work_dir} && {remote_cmd_str}"
    target = f"{user}@{host}" if user is not None else host
    ssh_cmd = ["ssh", "-p", str(port), target, remote_cmd_str]
    proc = subprocess.run(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(
            f"ssh/remote_cmd exited with code {proc.returncode}\n"
            f"SSH Command: {' '.join(ssh_cmd)}\n"
            f"Output:\n{output}"
        )
    return output


def run_power_monitor(
    full_cmd, total_duration, valid_duration, device, user, host, port
):
    print(
        f"Measuring power for {total_duration}s and take {valid_duration}s in the middle..."
    )
    print(full_cmd)

    power_monitor_path = os.path.abspath(os.path.join(file_dir, "power_monitor.py"))
    if not os.path.exists(power_monitor_path):
        raise FileNotFoundError(f"Cannot find power_monitor at {power_monitor_path}")

    with open(power_monitor_path, "r", encoding="utf-8") as f:
        script_body = f.read()

    variables_header = f"""
FULL_CMD = {repr(full_cmd)}
VALID_START_TIME = {total_duration / 2 - valid_duration / 2}
VALID_DURATION = {valid_duration}
DEVICE = "{device}"
"""
    remote_script_source = variables_header + "\n" + script_body
    b64_script = base64.b64encode(remote_script_source.encode("utf-8")).decode("utf-8")
    shell_pipeline = f"echo {b64_script} | base64 -d | python3"
    remote_cmd = ["bash", "-c", shell_pipeline]
    output = run_remote_command(user, host, port, remote_cmd).strip()

    try:
        avg_power_GPU = float(output.splitlines()[-2])
        avg_power_MEM = float(output.splitlines()[-1])
    except ValueError:
        raise RuntimeError(f"Could not parse power output. Received:\n{output}")

    return avg_power_GPU, avg_power_MEM
