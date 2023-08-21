import time
import random
import csv
from utils import get_random_sites
from selenium import webdriver

driver = webdriver.Chrome()
driver.maximize_window()


csv_file_path = "resource/majestic_million.csv"
SAVE_PATH = "test_img/"
RANDOM_SITES_COUNT = 200
MAX_ROWS_TO_READ = 10000


# def get_random_sites():
#     random_sites = []
#     with open(csv_file_path, "r", encoding="utf-8") as csv_file:
#         csv_reader = csv.reader(csv_file)
#         next(csv_reader)  # Skip the header row
#         random_rows = random.sample(list(csv_reader), min(RANDOM_SITES_COUNT, MAX_ROWS_TO_READ))
#         for row in random_rows:
#             domain = row[2]
#             random_sites.append(domain)
#     return random_sites

if __name__ == "__main__":
    random_websites = get_random_sites(csv_file_path, RANDOM_SITES_COUNT, MAX_ROWS_TO_READ)



    for random_website in random_websites:
        try:
            driver.get(  random_website)

            # Wait for the page to load
            time.sleep(3)

            # Capture and save a screenshot
            screenshot_path = SAVE_PATH+f'screenshot_{random_website}.png'
            driver.save_screenshot(screenshot_path)
            print(f'Screenshot taken from {random_website} and saved as {screenshot_path}')

            # Wait before the next visit
            time.sleep(random.randint(5, 15))
        except Exception as e:
            print(f"Error occurred while accessing {random_website}: {str(e)}")


    driver.quit()



