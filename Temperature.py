import gpustat
from datetime import datetime

class GPUStatus:
    def __init__(self):
        self.gpu_stats = None

    def update_gpu_stats(self):
        """Updates the GPU stats by querying the system."""
        self.gpu_stats = gpustat.GPUStatCollection.new_query()

    def get_gpu_temp(self):
        """Prints the GPU temperature for each GPU."""
        if self.gpu_stats is None:
            print("GPU stats not updated. Please call 'update_gpu_stats()' first.")
            return

        for gpu in self.gpu_stats:
            print(f"GPU {gpu.index}: {gpu.name} | Temperature: {gpu.temperature}°C")
    def Get_gpu_temperatures(self):
        temps=[]
        for gpu in self.gpu_stats:
            #print(f"GPU {gpu.index}: {gpu.name} | Temperature: {gpu.temperature}°C")
            if gpu.temperature > 75:
                print(f"Warning: GPU {gpu.index} ({gpu.name}) temperature is high! {gpu.temperature}°C at ",str(datetime.now()))
            temps.append(gpu.temperature)
        return temps


# Example of usage:
# gpu_status = GPUStatus()
# gpu_status.update_gpu_stats()  # Update GPU stats
# gpu_status.get_gpu_temp()      # Print GPU temperatures
