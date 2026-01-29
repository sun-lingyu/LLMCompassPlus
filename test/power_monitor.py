# power_monitor.py
import subprocess
import sys
import time

if "FULL_CMD" not in globals():  # to get rid of compiler complains
    FULL_CMD = ""
    VALID_START_TIME = 0
    VALID_DURATION = 1
    DEVICE = ""

assert DEVICE in ("Orin", "Thor")
if DEVICE == "Orin":
    VOLT_PATH_GPU = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input"
    CURR_PATH_GPU = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input"
    VOLT_PATH_VDDQ = "/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon2/in2_input"
    CURR_PATH_VDDQ = "/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon2/curr2_input"
elif DEVICE == "Thor":
    VOLT_PATH_GPU = "/sys/bus/i2c/drivers/ina3221/2-0040/hwmon/hwmon4/in1_input"
    CURR_PATH_GPU = "/sys/bus/i2c/drivers/ina3221/2-0040/hwmon/hwmon4/curr1_input"
    VOLT_PATH_VDDQ = "/sys/bus/i2c/drivers/ina3221/2-0040/hwmon/hwmon4/in3_input"
    CURR_PATH_VDDQ = "/sys/bus/i2c/drivers/ina3221/2-0040/hwmon/hwmon4/curr3_input"
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
                mV_VDD_GPU_SOC = int(fv.read().strip())
            with open(CURR_PATH_GPU, "r") as fc:
                mA_VDD_GPU_SOC = int(fc.read().strip())
            with open(VOLT_PATH_VDDQ, "r") as fv:
                mV_VDDQ_VDD2_1V8AO = int(fv.read().strip())
            with open(CURR_PATH_VDDQ, "r") as fc:
                mA_VDDQ_VDD2_1V8AO = int(fc.read().strip())

            power_VDD_GPU_SOC = (mV_VDD_GPU_SOC / 1000.0) * (mA_VDD_GPU_SOC / 1000.0)
            power_VDDQ_VDD2_1V8AO = (mV_VDDQ_VDD2_1V8AO / 1000.0) * (
                mA_VDDQ_VDD2_1V8AO / 1000.0
            )
            samples.append((now, power_VDD_GPU_SOC, power_VDDQ_VDD2_1V8AO))
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
    avg_power_VDD_GPU_SOC = sum([p[0] for p in valid_samples]) / len(valid_samples)
    avg_power_VDDQ_VDD2_1V8AO = sum([p[1] for p in valid_samples]) / len(valid_samples)
    print(avg_power_VDD_GPU_SOC)
    print(avg_power_VDDQ_VDD2_1V8AO)
else:
    print(0.0)
    print(0.0)
