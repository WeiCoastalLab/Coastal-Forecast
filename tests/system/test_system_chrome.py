from time import sleep

from selenium import webdriver
import unittest

from coastal_forecast import app


class TestAppChrome(unittest.TestCase):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(executable_path='chromedriver', options=options)

    @classmethod
    def setUpClass(cls) -> None:
        cls.driver.implicitly_wait(10)
        cls.driver.maximize_window()

    def test_app_chrome(self):
        self.driver.get('http://127.0.0.1:5000/')
        sleep(5)
        self.driver.get(self.driver.find_element_by_id('canaveral').get_attribute("href"))
        sleep(5)
        self.driver.get(self.driver.find_element_by_id('grays_reef').get_attribute("href"))
        sleep(5)
        self.driver.get(self.driver.find_element_by_id('frying_pan').get_attribute("href"))
        sleep(5)
        self.driver.get(self.driver.find_element_by_id('boston').get_attribute("href"))
        sleep(5)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.driver.close()
        cls.driver.quit()


if __name__ == "__main__":
    unittest.main()
