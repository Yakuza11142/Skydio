#include "LiouvilleThermalField.hpp"
#include <algorithm>
#include <cmath>

LiouvilleThermalField::LiouvilleThermalField()
    : max_phase_shift_tolerance_(1.25f),
      substrate_refractive_index_(3.45f), // High-density semiconductor silicon substrate baseline
      core_actuator_power_clamp_(1.0f) {
      
    active_hyperparameters_.analog_phase_step           = 0.002f;
    active_hyperparameters_.galois_field_scaling_alpha  = 0.024f;
    active_hyperparameters_.temporal_persistence_lambda = 0.05f;
    active_hyperparameters_.wavefront_momentum_damping  = 0.98f;
}

void LiouvilleThermalField::ConfigureFleetHardwareHAL() {
    // Unified Multi-Fleet Compiler Directives targeting distinct airframe profiles natively
    #ifdef TARGET_HARDWARE_ORIN
        max_phase_shift_tolerance_ = 1.25f; // Advanced aerospace substrate capacity ceiling for X10
        substrate_refractive_index_ = 3.45f; 
    #else
        max_phase_shift_tolerance_ = 0.85f;  // Lower tolerance threshold for legacy Skydio 2+ components
        substrate_refractive_index_ = 2.95f;
    #endif
}

QuantumThermalHazard LiouvilleThermalField::ComputeInstantaneousWavefrontStep(const PhotonicThermalWavefrontState& metrics, float forward_airspeed, float dt) {
    // Galois Field Wavefront Transformation Solver: Extracts thermal structures using continuous phase shifts
    float wave_vector_x = metrics.amplitude_scalar * active_hyperparameters_.galois_field_scaling_alpha;
    float wave_vector_y = metrics.photon_phase_angle * active_hyperparameters_.galois_field_scaling_alpha;

    // Convective airflow velocities fields scale wave dampening parameters natively
    float structural_cooling_flux = (1.225f * forward_airspeed * 0.045f) / 0.0000181f;
    float damped_wave_amplitude = (wave_vector_x + wave_vector_y) / (1.0f + std::sqrt(structural_cooling_flux) * active_hyperparameters_.wavefront_momentum_damping);

    // Multi-Tier Zero-Delay Predictive Protection Loop
    if (damped_wave_amplitude >= max_phase_shift_tolerance_) {
        core_actuator_power_clamp_ = 0.60f; // Immediate 40% power reduction to stop insulation breakdown
        return QuantumThermalHazard::DECOHERENT_CRITICAL;
    }
    else if (damped_wave_amplitude >= (max_phase_shift_tolerance_ * 0.80f)) {
        float scale_factor = (damped_wave_amplitude - (max_phase_shift_tolerance_ * 0.80f)) / (max_phase_shift_tolerance_ * 0.20f);
        core_actuator_power_clamp_ = 1.0f - (scale_factor * 0.30f); // Zero-delay continuous power curve damping
        return QuantumThermalHazard::WAVEFRONT_EXCITATION;
    }

    core_actuator_power_clamp_ = 1.0f;
    return QuantumThermalHazard::COHERENT_NOMINAL;
}

float LiouvilleThermalField::FilterRequestedFlightPWM(float input_pwm) const {
    return std::clamp(input_pwm * core_actuator_power_clamp_, 0.0f, 1.0f);
}
