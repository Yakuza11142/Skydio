#include "../src/mod_2_thermal_safety/include/LiouvilleThermalField.hpp"
#include <cassert>

void RunQuantumThermalClampingAssertion() {
    LiouvilleThermalField thermal_engine;
    thermal_engine.ConfigureFleetHardwareHAL();

    AtomicESCMetrics simulated_stress_packet{};
    simulated_stress_packet.phase_current_amperes = 55.0f; // High-current draw simulating aggressive acceleration
    simulated_stress_packet.magnetic_flux_webers = 0.015f;
    simulated_stress_packet.rotor_angular_velocity_rad_sec = 1100.0f;
    simulated_stress_packet.ambient_inverter_temperature_kelvin = 350.15f; // Highly excited ambient environment

    QuantumThermalHazard state = QuantumThermalHazard::COHERENT_NOMINAL;
    
    // Simulate continuous 100Hz execution steps across 10 sequential seconds of sustained overload stress
    constexpr float dt = 0.01f;
    for (int step = 0; step < 1000; ++step) {
        state = thermal_engine.ComputeInstantaneousWavefrontStep(simulated_stress_packet, 0.0f, dt);
    }

    // Hard Asset Condition: The engine MUST isolate extreme entropy loads and trigger the protective clamp
    assert(state == QuantumThermalHazard::DECOHERENT_CRITICAL);
    float baseline_unclamped_pwm = 1.0f;
    float secured_pwm_output = thermal_engine.FilterRequestedFlightPWM(baseline_unclamped_pwm);
    
    // Confirm the current clamp dynamically attenuated the duty cycle to a safe 60% baseline configuration profile
    assert(secured_pwm_output == 0.60f);
}

int main() {
    RunQuantumThermalClampingAssertion();
    return 0;
}
