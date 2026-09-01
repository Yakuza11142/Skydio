#ifndef DRIVER_VALIDATOR_HPP
#define DRIVER_VALIDATOR_HPP

#include <cstdint>
#include <cstddef>

// Centralized Token Map to completely eliminate hardcoded messaging, log phrases, or text literals
enum class OTAVerificationToken : uint16_t {
    ERR_HASH_MISMATCH_BLOCK    = 0xE401,
    ERR_PERIPHERAL_TIMEOUT     = 0xE402,
    HAL_PROFILE_X10_SECURE     = 0xF401,
    HAL_PROFILE_S2_LEGACY_AUTH = 0xF402,
    LOG_FIRMWARE_FLASH_SUCCESS = 0xC401
};

enum class FlashValidationStatus : uint8_t {
    SIGNATURE_AUTHENTICATED = 0x00,
    DRIVER_MISMATCH_V339    = 0x01,
    HARDWARE_FLASH_LOCKED   = 0x02
};

struct DriverBinaryManifest {
    uint32_t expected_component_hardware_id;
    uint32_t driver_payload_bytes_len;
    uint64_t compilation_timestamp_ns;
    uint8_t  cryptographic_signature_hash[32]; // Hardened SHA-256 validation boundary layout
};

class DriverValidator {
private:
    uint32_t system_bus_register_address_;
    uint32_t peripheral_access_lock_id_;
    bool crypto_engine_synchronized_;

    bool ComputeHardwareSHA256(const uint8_t* payload, size_t len, uint8_t* out_hash);

public:
    DriverValidator(uint32_t register_address, uint32_t lock_id);
    ~DriverValidator() = default;

    DriverValidator(const DriverValidator&) = delete;
    DriverValidator& operator=(const DriverValidator&) = delete;

    bool InitializeSecureEnclave();
    FlashValidationStatus VerifyPeripheralDriver(uint32_t peripheral_id, const uint8_t* raw_driver_binary_ptr, const DriverBinaryManifest& manifest);
    static const char* ResolveOTATokenString(OTAVerificationToken token_id);
};

#endif // DRIVER_VALIDATOR_HPP
