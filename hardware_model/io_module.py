class IOModule:
    def __init__(self, bandwidth, bandwidth_efficiency, latency_cycles):
        self.bandwidth = bandwidth  # bytes per second
        self.bandwidth_efficiency = bandwidth_efficiency
        self.latency_cycles = latency_cycles  # cycles
