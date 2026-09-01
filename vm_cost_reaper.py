import time
from typing import List, Dict, Any

class VMCostReaper:
    def __init__(self, idle_timeout_threshold_seconds: int):
        self.timeout_limit = idle_timeout_threshold_seconds
        self.terminated_node_counter = 0

    def query_active_cloud_simulation_nodes(self) -> List[Dict[str, Any]]:
        # Simulation profiles gathered from the active cluster (Mock structure representing AWS/GCP API responses)
        return [
            {"instance_id": "i-09f1a23e4b51c67d8", "idle_duration_sec": 7200, "owner_token": "dev_node_01"},
            {"instance_id": "i-012c345d6e78f90ab", "idle_duration_sec": 300,  "owner_token": "dev_node_02"},
            {"instance_id": "i-0fa1b23c4d56e78f9", "idle_duration_sec": 28800, "owner_token": "zombie_simulation_node"}
        ]

    def execute_infrastructure_sweep(self) -> int:
        """Audits active compute runtimes and purges stranded simulation instances to save infrastructure bills."""
        active_instances = self.query_active_cloud_simulation_nodes()
        self.terminated_node_counter = 0

        for instance in active_instances:
            if instance["idle_duration_sec"] >= self.timeout_limit:
                # In final orchestration layers, this triggers the direct boto3/GCP API call 
                # to securely terminate the specific virtual machine compute thread instantly
                print(f"[REAPER_ACTION] Terminating zombie instance: {instance['instance_id']}")
                self.terminated_node_counter += 1

        return self.terminated_node_counter

if __name__ == "__main__":
    # Standard engineering sweep protocol: Automatically sweep and reap everything left idle for > 1 hour
    reaper = VMCostReaper(idle_timeout_threshold_seconds=3600)
    reaper.execute_infrastructure_sweep()
