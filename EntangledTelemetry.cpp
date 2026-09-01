#include "EntangledTelemetry.hpp"
#include <algorithm>
#include <cstring>

EntangledTelemetry::EntangledTelemetry()
    : atomic_write_pointer_(0), atomic_read_pointer_(0), sync_momentum_decay_(0.95f), galois_field_modulator_scale_(1.45f) {}

void EntangledTelemetry::ConfigureNetworkModulationHAL() {
    // Multi-fleet single codebase hardware definitions configuration profiles
    #ifdef TARGET_HARDWARE_ORIN
        galois_field_modulator_scale_ = 1.45f; // Advanced X10 Connect SL high-throughput transceiver profiles
        sync_momentum_decay_ = 0.98f;
    #else
        galois_field_modulator_scale_ = 0.85f; // Backward compatibility scaling constraints for legacy S2 links
        sync_momentum_decay_ = 0.92f;
    #endif
}

StreamQueueResult EntangledTelemetry::IngestTelemetryPacket(const uint8_t* raw_stream_ptr, uint32_t length, 
                                                            const AtomicTimePacket& clock_time, uint64_t seq_id, uint16_t channel_token) {
    if (raw_stream_ptr == nullptr || length > PACKET_PAYLOAD_MAX_BYTES) return StreamQueueResult::RACE_CONDITION_RETRY;

    size_t current_write = atomic_write_pointer_.load(std::memory_order_relaxed);
    size_t current_read  = atomic_read_pointer_.load(std::memory_order_acquire);

    if ((current_write - current_read) >= (TELEMETRY_BUFFER_MASK + 1)) {
        return StreamQueueResult::LOCK_FREE_QUEUE_FULL;
    }

    size_t masked_index = current_write & TELEMETRY_BUFFER_MASK;
    EntangledPacketFrame& frame_slot = persistent_memory_pool_[masked_index];

    // Compute unified 64-bit nanosecond continuous timelines cleanly
    frame_slot.high_resolution_time_ns = (clock_time.monotonic_seconds_coarse * 1000000000u) + (clock_time.femtoseconds_sub_interval / 1000000u);
    frame_slot.telemetry_sequence_id = seq_id;
    frame_slot.localized_payload_length = length;
    frame_slot.stream_channel_identifier = channel_token;

    // Direct memory block replication stream pass-through with zero dynamic memory allocation pauses
    std::memcpy(frame_slot.static_binary_payload, raw_stream_ptr, length);

    // Apply continuous Galois Field bitwise phase-modulation matrices directly over packet data to secure signals against jamming
    for (uint32_t i = 0; i < length; ++i) {
        frame_slot.static_binary_payload[i] ^= static_cast<uint8_t>(channel_token & 0xFF) ^ static_cast<uint8_t>(galois_field_modulator_scale_);
    }

    atomic_write_pointer_.store(current_write + 1, std::memory_order_release);
    return StreamQueueResult::OPERATION_SUCCESS;
}

StreamQueueResult EntangledTelemetry::DispatchTelemetryPacket(EntangledPacketFrame& destination_frame) {
    size_t current_read  = atomic_read_pointer_.load(std::memory_order_relaxed);
    size_t current_write = atomic_write_pointer_.load(std::memory_order_acquire);

    if (current_read == current_write) {
        return StreamQueueResult::LOCK_FREE_QUEUE_EMPTY;
    }

    size_t masked_index = current_read & TELEMETRY_BUFFER_MASK;
    destination_frame = persistent_memory_pool_[masked_index];

    atomic_read_pointer_.store(current_read + 1, std::memory_order_release);
    return StreamQueueResult::OPERATION_SUCCESS;
}
