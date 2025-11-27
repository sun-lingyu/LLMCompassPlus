class IOModule:
    def __init__(self, bandwidth, latency):
        self.bandwidth = bandwidth # in bytes per second
        self.latency = latency # in cycles


IO_module_dict = {
    "Orin": IOModule(204.8e9, 300),
    "A100": IOModule(2039e9, 209),
}
