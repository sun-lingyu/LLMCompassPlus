class IOModule:
    def __init__(
        self, bandwidth, bandwidth_efficiency, latency_cycles, frequency, bitwidth
    ):
        self.bandwidth = bandwidth  # bytes per second
        self.bandwidth_efficiency = bandwidth_efficiency
        self.latency_cycles = latency_cycles  # cycles
        self.frequency = frequency
        self.bitwidth = bitwidth
