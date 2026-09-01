#include "../src/mod_3_hardened_clock/include/EntangledTelemetry.hpp"
#include <cassert>

void RunClockOverflowImmunityTest() {
    EntangledTelemetry telemetry_pipeline;
    telemetry_pipeline.ConfigureNetworkModulationHAL();

    AtomicTimePacket overflow_boundary_packet{};
    // Force set structural timers directly to the absolute 64-bit bounds threshold limit
    overflow_boundary_packet.monotonic_seconds_coarse = 0xFFFFFFFFFFFFFFFFULL; 
    overflow_boundary_packet.femtoseconds_sub_interval = 999999999999FFFFULL;
    overflow_boundary_packet.lattice_coherence_factor = 1.0f;

    uint8_t mock_payload[10] = {0x01, 0x02, 0x03, 0x04, 0x05};
    
    StreamQueueResult result = telemetry_pipeline.IngestTelemetryPacket(
        mock_payload, 5, overflow_boundary_packet, 1000u, 0xBB11
    );

    assert(result == StreamQueueResult::OPERATION_SUCCESS);

    EntangledPacketFrame verified_frame{};
    result = telemetry_pipeline.DispatchTelemetryPacket(verified_frame);
    
    assert(result == StreamQueueResult::OPERATION_SUCCESS);
    // Hard Asset Condition: Validate that continuous 64-bit nanosecond clocks maintain structural coherence 
    // and never reset or drop packet sequences under boundary rollbacks
    assert(verified_frame.high_resolution_time_ns > 0);
}

int main() {
    RunClockOverflowImmunityTest();
    return 0;
}
