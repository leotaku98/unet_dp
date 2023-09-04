from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils import *

csv_file_path = "resource/majestic_million.csv"
SAVE_PATH = "test_img/"
RANDOM_SITES_COUNT = 50
MAX_ROWS_TO_READ = 10000
TIMEOUT_SECONDS = 10



if __name__ == "__main__":
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    driver.implicitly_wait(5)
    random_websites = get_random_sites(csv_file_path, RANDOM_SITES_COUNT, MAX_ROWS_TO_READ)

    index = 0
    #random_websites = ["www.quooker.nl/"]
    for random_website in random_websites:
        index += 1
        print(index, random_website)
        try:
            driver.get("http://" + random_website)
            WebDriverWait(driver, TIMEOUT_SECONDS).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            icons = driver.find_elements(By.XPATH, "//a ")

            for icon in icons:
                if icon.text and icon.text.isascii():
                    href = icon.get_attribute("href")
                    if href is not None and not compare_similarity(icon.text, href, 0.8):
                        print(f"misdirection, you are out icon_text:{icon.text} - href:{href}")
                        with open("misdirection.txt", "a") as file:
                            file.write(f"{random_website} - icon_text:{icon.text} - href:{href}\n")
                        #break
                    else:
                        pass
                        print(f"{random_website} - icon_text:{icon.text} - href:{href}\n")
        except Exception as e:
            #continue
            print(f"Error occurred while accessing {random_website}: {str(e)}")

    driver.quit()
