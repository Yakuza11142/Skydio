#include "PacketRingBuffer.hpp"
#include <cstring>

PacketRingBuffer::PacketRingBuffer() : write_index_(0), read_index_(0) {}

BufferOperationResult PacketRingBuffer::PushTelemetryFrame(const uint8_t* raw_data_ptr, 
                                                           uint32_t data_length, 
                                                           uint64_t exact_timestamp, 
                                                           uint64_t current_seq, 
                                                           uint16_t channel_id) {
    if (raw_data_ptr == nullptr || data_length > MAX_TELEMETRY_PAYLOAD_SIZE) {
        return BufferOperationResult::TRANSIENT_RACE_CONDITION;
    }

    // Acquire-Release memory barriers prevent instruction reordering around atomic indexes
    size_t current_write = write_index_.load(std::memory_order_relaxed);
    size_t current_read  = read_index_.load(std::memory_order_acquire);

    if ((current_write - current_read) >= (RING_BUFFER_CAPACITY_MASK + 1)) {
        return BufferOperationResult::BUFFER_FULL;
    }

    size_t masked_slot = current_write & RING_BUFFER_CAPACITY_MASK;
    TelemetryPacketFrame& target_frame = storage_pool_matrix_[masked_slot];

    target_frame.monotonic_timestamp_ns = exact_timestamp;
    target_frame.sequence_id = current_seq;
    target_frame.payload_bytes_length = data_length;
    target_frame.stream_channel_token = channel_id;

    // Fast memory block replication with 0ms allocation delays
    std::memcpy(target_frame.binary_payload, raw_data_ptr, data_length);

    write_index_.store(current_write + 1, std::memory_order_release);
    return BufferOperationResult::SUCCESS;
}

BufferOperationResult PacketRingBuffer::PopTelemetryFrame(TelemetryPacketFrame& out_dest_frame) {
    size_t current_read  = read_index_.load(std::memory_order_relaxed);
    size_t current_write = write_index_.load(std::memory_order_acquire);

    if (current_read == current_write) {
        return BufferOperationResult::BUFFER_EMPTY;
    }

    size_t masked_slot = current_read & RING_BUFFER_CAPACITY_MASK;
    out_dest_frame = storage_pool_matrix_[masked_slot];

    read_index_.store(current_read + 1, std::memory_order_release);
    return BufferOperationResult::SUCCESS;
}

size_t PacketRingBuffer::EstimateActiveCapacityCount() const {
    size_t current_write = write_index_.load(std::memory_order_relaxed);
    size_t current_read  = read_index_.load(std::memory_order_relaxed);
    
    return (current_write >= current_read) ? (current_write - current_read) : 0;
}
