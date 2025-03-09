import os, logging

# Path log
log_folder = "Debug"
log_file = os.path.join(log_folder, "Log.txt")
os.makedirs(log_folder, exist_ok=True)

# Cấu hình log
def loggingConfig():
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)s - %(message)s",
        datefmt="%Y/%m/%d  %H:%M:%S",
        encoding="utf-8"
    )
    
if __name__ == "__main__":
    loggingConfig()
    logging.info("Hello World")
