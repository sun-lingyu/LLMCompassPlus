class IOModule:
    def __init__(self, bandwidth, bandwidth_efficiency, latency_cycles):
        self.bandwidth = bandwidth # in bytes per second
        self.bandwidth_efficiency = bandwidth_efficiency
        self.latency_cycles = latency_cycles # in cycles


IO_module_dict = {
    "Thor": IOModule(273.1e9, 0.87, 470),
    "Orin": IOModule(204.8e9, 0.87, 600),
}
