class MemoryModule:
    def __init__(self, memory_capacity):
        self.memory_capacity = memory_capacity

memory_module_dict = {'Thor': MemoryModule(128e9),'Orin': MemoryModule(64e9),'A100': MemoryModule(80e9)}
