#ifndef MONOTONIC_CLOCK_HAL_HPP
#define MONOTONIC_CLOCK_HAL_HPP

#include <cstdint>

// Centralized Token Map to completely eliminate hardcoded messaging, log phrases, or text literals
enum class ClockSystemToken : uint16_t {
    ERR_LATTICE_DECOHERENCE    = 0xE301,
    ERR_TIME_DILATION_OVERFLOW = 0xE302,
    HAL_PROFILE_X10_ORIN_CLOCK = 0xF301,
    HAL_PROFILE_S2_TX2_CLOCK   = 0xF302,
    LOG_CLOCK_SYNC_VALIDATED   = 0xC301
};

enum class ClockTelemetryStatus : uint8_t {
    STABLE_ATOMIC_TIME    = 0x00,
    LATTICE_PHASE_BUSY    = 0x01,
    COHERENCE_FRAME_FAULT = 0x02
};

struct AtomicTimePacket {
    uint64_t femtoseconds_sub_interval; // Microsecond-shattering quantum clock precision
    uint64_t monotonic_seconds_coarse;  // True 64-bit counter completely immune to rollover bugs
    float lattice_coherence_factor;
    float relative_time_dilation_delta; 
};

class MonotonicClockHAL {
private:
    uint32_t laser_waveguide_channel_;
    uint32_t atomic_trap_lane_id_;
    bool clock_hardware_locked_;

public:
    MonotonicClockHAL(uint32_t wave_channel, uint32_t trap_lane);
    ~MonotonicClockHAL() = default;

    MonotonicClockHAL(const MonotonicClockHAL&) = delete;
    MonotonicClockHAL& operator=(const MonotonicClockHAL&) = delete;

    bool SynchronizeAtomicLattice();
    ClockTelemetryStatus ReadPhotonicTimeState(AtomicTimePacket& out_time);
    static const char* ResolveClockTokenString(ClockSystemToken token_id);
};

#endif // MONOTONIC_CLOCK_HAL_HPP
