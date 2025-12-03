class IOModule:
    def __init__(self, bandwidth, latency_cycles):
        self.bandwidth = bandwidth # in bytes per second
        self.latency_cycles = latency_cycles # in cycles


IO_module_dict = {
    "Orin": IOModule(204.8e9, 300),
    "A100": IOModule(2039e9, 209),
}
