import os
import requests
import time


URLS = [
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_air_passengers.csv",
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_retail_sales.csv",
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_R.csv",
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_R_outliers1.csv",
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_R_outliers2.csv",
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_peyton_manning.csv",
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_yosemite_temps.csv"
]


def get_data(url):
    filename = os.path.basename(url)
    filename = os.path.join("data/input", filename)
    if os.path.exists(filename):
        print(f"exists: {filename}")
        return
    
    response = requests.get(url)
    time.sleep(1)
    
    if response.status_code == 200:
        with open(filename, "w") as f:
            f.write(response.text)
        print(f"got: {filename}")
    else:
        print("DOWNLOAD ERROR")


if __name__ == "__main__":
    for url in URLS:
        get_data(url)
