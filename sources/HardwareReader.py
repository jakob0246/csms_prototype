import psutil
import cpuinfo

import numpy as np

from enum import Enum


class ByteType(Enum):
    byte = 0
    kilobyte = 1
    megabyte = 2
    gigabyte = 3
    terabyte = 4


class Byte:
    size: int  # save size in byte

    # attribute size needs to be in byte, if Byte is constructed
    def __init__(self, size: int, type):
        if type == ByteType.byte:
            Byte.size = size
        else:
            Byte.size = size * (type.value * (10 ** 3))

    def get_size(self, byte_type, decimals: int = None):
        if decimals is None:
            return Byte.size / (10 ** (byte_type.value * 3))
        else:
            return np.around(Byte.size / (10 ** (byte_type.value * 3)), decimals)


def get_hardware_specs():
    hardware_specs = {
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram": Byte(psutil.virtual_memory().total, ByteType.byte).get_size(ByteType.gigabyte, 2),
        "cache_l3": None,
        "cache_l2": Byte(int(cpuinfo.get_cpu_info()["l2_cache_size"]), ByteType.kilobyte).get_size(ByteType.kilobyte, 1),
    }

    return hardware_specs
