# power_monitor.py
import glob
import subprocess
import sys
import time


def get_glob_path(path):
    res = glob.glob(path)
    return res[0] if res else None


if "FULL_CMD" not in globals():  # to get rid of compiler complains
    FULL_CMD = ""
    VALID_START_TIME = 0
    VALID_DURATION = 1
    DEVICE = ""

assert DEVICE in ("Orin", "Thor")
if DEVICE == "Orin":
    VOLT_PATH_GPU = get_glob_path(
        "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/in1_input"
    )
    CURR_PATH_GPU = get_glob_path(
        "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/curr1_input"
    )
    VOLT_PATH_MEM = get_glob_path(
        "/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon*/in2_input"
    )
    CURR_PATH_MEM = get_glob_path(
        "/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon*/curr2_input"
    )
elif DEVICE == "Thor":
    VOLT_PATH_GPU = get_glob_path(
        "/sys/bus/i2c/drivers/ina3221/2-0040/hwmon/hwmon*/in1_input"
    )
    CURR_PATH_GPU = get_glob_path(
        "/sys/bus/i2c/drivers/ina3221/2-0040/hwmon/hwmon*/curr1_input"
    )
    VOLT_PATH_MEM = get_glob_path(
        "/sys/bus/i2c/drivers/ina3221/2-0040/hwmon/hwmon*/in3_input"
    )
    CURR_PATH_MEM = get_glob_path(
        "/sys/bus/i2c/drivers/ina3221/2-0040/hwmon/hwmon*/curr3_input"
    )
else:
    assert False, f"illegal {DEVICE}"


proc = subprocess.Popen(
    FULL_CMD, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)

samples = []  # (timestamp, power_watts_gpu, power_watts_vddq)

try:
    while proc.poll() is None:
        try:
            now = time.time()
            with open(VOLT_PATH_GPU, "r") as fv:
                mV_GPU = int(fv.read().strip())
            with open(CURR_PATH_GPU, "r") as fc:
                mA_GPU = int(fc.read().strip())
            with open(VOLT_PATH_MEM, "r") as fv:
                mV_MEM = int(fv.read().strip())
            with open(CURR_PATH_MEM, "r") as fc:
                mA_MEM = int(fc.read().strip())

            power_GPU = (mV_GPU / 1000.0) * (mA_GPU / 1000.0)
            power_MEM = (mV_MEM / 1000.0) * (mA_MEM / 1000.0)
            samples.append((now, power_GPU, power_MEM))
        except Exception:
            pass  # ignore temporary read errors

        time.sleep(0.05)

finally:
    if proc.poll() is None:
        proc.terminate()

if not samples:
    print(0.0)
    print(0.0)
    sys.exit(0)

end_time = samples[-1][0]
start_time = samples[0][0]

valid_start_time = start_time + VALID_START_TIME
valid_end_time = valid_start_time + VALID_DURATION

valid_samples = [
    (p1, p2) for (t, p1, p2) in samples if t >= valid_start_time and t <= valid_end_time
]

if valid_samples:
    avg_power_GPU = sum([p[0] for p in valid_samples]) / len(valid_samples)
    avg_power_MEM = sum([p[1] for p in valid_samples]) / len(valid_samples)
    print(avg_power_GPU)
    print(avg_power_MEM)
else:
    print(0.0)
    print(0.0)
