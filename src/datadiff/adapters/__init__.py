"""Adapter capability, lowering, execution, plan, and native-hook boundary."""

from datadiff.backends.spi import AdapterSPIManifest
from datadiff.targets import TargetSpec

__all__ = ["AdapterSPIManifest", "TargetSpec"]
