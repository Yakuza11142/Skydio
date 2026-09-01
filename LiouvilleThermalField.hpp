#ifndef LIOUVILLE_THERMAL_FIELD_HPP
#define LIOUVILLE_THERMAL_FIELD_HPP

#include "QuantumThermalHAL.hpp"

enum class QuantumThermalHazard : uint8_t {
    COHERENT_NOMINAL   = 0x00,
    WAVEFRONT_EXCITATION = 0x01,
    DECOHERENT_CRITICAL = 0x02
};

struct ZeroDelayThermalHyperparameters {
    float analog_phase_step;
    float galois_field_scaling_alpha;
    float temporal_persistence_lambda;
    float wavefront_momentum_damping;
};

class LiouvilleThermalField {
private:
    ZeroDelayThermalHyperparameters active_hyperparameters_{};
    float max_phase_shift_tolerance_;
    float substrate_refractive_index_;
    float core_actuator_power_clamp_;

public:
    LiouvilleThermalField();
    ~LiouvilleThermalField() = default;

    void ConfigureFleetHardwareHAL();
    QuantumThermalHazard ComputeInstantaneousWavefrontStep(const PhotonicThermalWavefrontState& metrics, float forward_airspeed, float dt);
    
    [[nodiscard]] float FilterRequestedFlightPWM(float input_pwm) const;
    [[nodiscard]] float GetActivePowerClamp() const { return core_actuator_power_clamp_; }
};

#endif // LIOUVILLE_THERMAL_FIELD_HPP
