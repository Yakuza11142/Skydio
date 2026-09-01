# Monorepo Module Configuration Entrypoint
# Centralized registry definition layer to completely eliminate hardcoded configuration parameters

__version__ = "2.0.0"
__module_name__ = "skydio_x10_cloud_devops_harness"

# Abstracted core status tokens mapping to system registers
SYS_TOKEN_REAPER_ACTIVE = 0x501
SYS_TOKEN_HIL_CONNECTED = 0x502
SYS_TOKEN_TEST_PASSED   = 0x503
SYS_TOKEN_TEST_FAILED   = 0x504
