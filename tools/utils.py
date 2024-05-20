from torch import Tensor
from enum import Enum

class Unit(Enum):
  BYTE   = 'BYTE'
  KB     = 'KB'
  MB     = 'MB'
  GB     = 'GB'
  TB     = 'TB'

def get_parameter_nums(parameters: dict[Tensor], unit: Unit | str = Unit.MB) -> int:
  total_parameter_size = 0
  if type(unit) == str: unit = unit.upper()
  
  for parameter in parameters:
    total_parameter_size += parameter.nelement()
    
  total_parameter_size *= parameter.dtype.itemsize
  
  if unit == Unit.BYTE:
    total_parameter_size = total_parameter_size
  elif unit == Unit.KB:
    total_parameter_size = total_parameter_size / 1024
  elif unit == Unit.MB:
    total_parameter_size = total_parameter_size / 1024 / 1024
  elif unit == Unit.GB:
    total_parameter_size = total_parameter_size / 1024 / 1024 / 1024
  elif unit == Unit.TB:
    total_parameter_size = total_parameter_size / 1024 / 1024 / 1024 / 1024

  return total_parameter_size
