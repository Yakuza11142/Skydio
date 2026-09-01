#ifndef REAL_TIME_WATCHDOG_HPP
#define REAL_TIME_WATCHDOG_HPP

#include <cstdint>

struct CoreFlightSchedulerState {
    uint64_t primary_loop_execution_counter;
    uint32_t scheduler_interrupt_lock_flag;
    uint8_t  system_critical_freeze_detected;
};

class RealTimeWatchdog {
private:
    uint64_t max_allowed_idle_ticks_;
    uint64_t historical_tick_counter_;
    uint64_t stall_duration_accumulator_;
    bool watchdog_armed_;

public:
    RealTimeWatchdog();
    ~RealTimeWatchdog() = default;

    RealTimeWatchdog(const RealTimeWatchdog&) = delete;
    RealTimeWatchdog& operator=(const RealTimeWatchdog&) = delete;

    void ArmWatchdogSystem(uint64_t timeout_ticks);
    void EvaluateFlightLoopHealth(CoreFlightSchedulerState* shared_state_ptr, uint32_t* motor_hardware_registers, uint32_t total_motors);
};

#endif // REAL_TIME_WATCHDOG_HPP
