# Import
import os, logging, psutil, platform, subprocess

# Import test
if __name__ == "__main__":
    from Debug.Logging import loggingConfig
    loggingConfig()

# Cấu hình
app_name = "msedge.exe"

def check_edge_running(app_name):
    logging.info("Kiểm tra Edge...")
    
    if platform.system() == "Windows":
        return any(app_name.lower() in process.info["name"].lower() for process in psutil.process_iter(attrs=["name"]))
    
    else:
        try:
            output = subprocess.check_output(["pgrep", "-f", "microsoft-edge"])
            return bool(output.strip())
        except subprocess.CalledProcessError:
            return False

def open_edge():
	if not check_edge_running(app_name):
		logging.info("Edge chưa chạy, đang mở edge...")
  