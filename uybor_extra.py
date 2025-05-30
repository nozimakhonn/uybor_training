from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
import pandas as pd
import time

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

flat_titles = []
squares = []
prices = []
rooms = []
floors = []
b_floors = []
renovations = []
materials = []
addresses = []
lifts = []
bathrooms = []
wash_machs = []
tvs = []
microwave_ovens = []
playgrounds = []
furnitures = []
refrigerators = []
air_conditioners = []
wifis = []
cables = []
securities = []
video_surveillances = []
parking_spaces = []

next_url = "https://uybor.uz/listings?category__eq=7&isNewBuilding__eq=false&page=1"
driver.get(next_url)
count = 1
# def coordinates(address):
#     client = ymaps.Geocode()
for i in range(98):
    time.sleep(5)  # Wait 5 seconds before each request
    driver.get(next_url)

    try:
        next_url = f"https://uybor.uz/listings?category__eq=7&isNewBuilding__eq=false&page={i + 1}"
        print(next_url)
    except:
        print('No next page')
        break

    try:
        flat_urls = WebDriverWait(driver, 20).until(  # Increased timeout
            expected_conditions.presence_of_all_elements_located(
                (By.XPATH, "//a[@class='MuiBox-root mui-style-1vssrzj']")
            )
        )
    except:
        break
    flat_links = []
    for url in flat_urls:
        link = url.get_attribute("href")
        flat_links.append(link)
    print(flat_links)
    print(f'Number of flats: {len(flat_links)}')

    for link in flat_links:
        print(count)
        driver.get(link)
        try:
            title = WebDriverWait(driver, 10).until(expected_conditions.presence_of_element_located((By.XPATH,
                                                                                                     "//h1[@class='MuiTypography-root MuiTypography-h2 mui-style-1tyknu']"))).text
        except:
            title = 'No title'
        flat_titles.append(title)
        print(title)

        try:
            price = driver.find_element(By.XPATH,
                                        "//div[@class='MuiTypography-root MuiTypography-h2 mui-style-86wpc3']").text
        except:
            price = 'No price'
        prices.append(price)
        print(f'Price: {price}')

        try:
            room = driver.find_element(By.XPATH,
                                       "//div[@class='MuiTypography-root MuiTypography-overline mui-style-1xqesu' and contains(text(), 'Комнат')]"
                                       "/following-sibling::div").text
        except:
            room = 'No room'
        rooms.append(room)
        print(f'Room: {room}')

        try:
            square = driver.find_element(By.XPATH,
                                         "//div[@class='MuiTypography-root MuiTypography-overline mui-style-1xqesu' and contains(text(), 'Площадь')]"
                                         "/following-sibling::div").text
        except:
            square = 'No square'
        squares.append(square)
        print(f'Square: {square}')

        try:
            floor = driver.find_element(By.XPATH,
                                        "//div[@class='MuiTypography-root MuiTypography-overline mui-style-1xqesu' and contains(text(), 'Этаж')]"
                                        "/following-sibling::div").text
        except:
            floor = 'No floor'

        parts = floor.split('/')

        floors.append(parts[0])
        if len(parts) > 1:
            b_floors.append(parts[1])
        else:
            b_floors.append('Unknown')

        print('Floor:', floor)

        try:
            renovation = driver.find_element(By.XPATH,
                                             "//div[@class='MuiTypography-root MuiTypography-overline mui-style-1xqesu' and contains(text(), 'Ремонт')]"
                                             "/following-sibling::div").text
        except:
            renovation = 'No renovation'
        renovations.append(renovation)
        print('Renovation:', renovation)


        try:
            material = driver.find_element(By.XPATH,
                                           "//div[@class='MuiTypography-root MuiTypography-overline mui-style-1xqesu' and contains(text(), 'Материал')]"
                                           "/following-sibling::div").text
        except:
            material = 'No material'
        materials.append(material)
        print('Material:', material)

        try:
            address = driver.find_element(By.XPATH,
                                          "//div[@class='MuiTypography-root MuiTypography-body2 mui-style-31fjox']").text
        except:
            address = 'No address'
        addresses.append(address)
        print('Address:', address)

        try:
            lift = driver.find_element(By.XPATH,
                                       "//div[@class='MuiTypography-root MuiTypography-body3 mui-style-xckitu' and contains(text(), 'Лифт')]").text
            lift = 1
        except:
            if int(floor.split('/')[1]) > 5:
                lift = 1
            else:
                lift = 0

        lifts.append(lift)
        print('Lift:', lift)

        try:
            bathroom = driver.find_element(By.XPATH,
                                           "//div[@class='MuiTypography-root MuiTypography-body3 mui-style-xckitu' and contains(text(), 'Санузел раздельный')]").text
            bathroom = 1
        except:
            bathroom = 0

        bathrooms.append(bathroom)
        print('Sanuzer:', bathroom)

        try:
            wash_mach = driver.find_element(By.XPATH,
                                            "//div[@class='MuiTypography-root MuiTypography-body3 mui-style-xckitu' and contains(text(), 'Стиральная машина')]").text
            wash_mach = 1
        except:
            wash_mach = 0

        wash_machs.append(wash_mach)
        print('Wash_mach:', wash_mach)

        try:
            tv = driver.find_element(By.XPATH,
                                     "//div[@class='MuiTypography-root MuiTypography-body3 mui-style-xckitu' and contains(text(), 'Телевизор')]").text
            tv = 1
        except:
            tv = 0

        tvs.append(tv)
        print('TV:', tv)

        try:
            playground = driver.find_element(By.XPATH,
                                             "//div[@class='MuiTypography-root MuiTypography-body3 mui-style-xckitu' and contains(text(), 'Детская площадка')]").text
            playground = 1
        except:
            playground = 0

        playgrounds.append(playground)
        print('Playground:', playground)

        try:
            furniture = driver.find_element(By.XPATH,
                                            "//div[@class='MuiTypography-root MuiTypography-body3 mui-style-xckitu' and contains(text(), 'Мебель')]").text
            furniture = 1
        except:
            furniture = 0

        furnitures.append(furniture)
        print('Furniture:', furniture)

        try:
            refrigerator = driver.find_element(By.XPATH,
                                               "//div[@class='MuiTypography-root MuiTypography-body3 mui-style-xckitu' and contains(text(), 'Холодильник')]").text
            refrigerator = 1
        except:
            refrigerator = 0

        refrigerators.append(refrigerator)
        print('Refrigerator:', refrigerator)

        try:
            microwave_oven = driver.find_element(By.XPATH,
                                                 "//div[@class='MuiTypography-root MuiTypography-body3 mui-style-xckitu' and contains(text(), 'Микроволновая печь')]").text
            microwave_oven = 1
        except:
            microwave_oven = 0

        microwave_ovens.append(microwave_oven)
        print('Microwave_oven:', microwave_oven)


        try:
            air_conditioner = driver.find_element(By.XPATH,
                                                  "//div[@class='MuiTypography-root MuiTypography-body3 mui-style-xckitu' and contains(text(), 'Кондиционер')]").text
            air_conditioner = 1
        except:
            air_conditioner = 0

        air_conditioners.append(air_conditioner)
        print('Air_conditioner:', air_conditioner)

        try:
            wifi = driver.find_element(By.XPATH,
                                       "//div[@class='MuiTypography-root MuiTypography-body3 mui-style-xckitu' and contains(text(), 'Интернет')]").text
            wifi = 1
        except:
            wifi = 0

        wifis.append(wifi)
        print('Wifi:', wifi)

        try:
            cable = driver.find_element(By.XPATH,
                                        "//div[@class='MuiTypography-root MuiTypography-body3 mui-style-xckitu' and contains(text(), 'Спутниковое/кабельное ТВ')]").text
            cable = 1
        except:
            cable = 0

        cables.append(cable)
        print('Cable:', cable)

        try:
            security = driver.find_element(By.XPATH,
                                           "//div[@class='MuiTypography-root MuiTypography-body3 mui-style-xckitu' and contains(text(), 'Охрана')]").text
            security = 1
        except:
            security = 0

        securities.append(security)
        print('Security:', security)

        try:
            video_surveillance = driver.find_element(By.XPATH,
                                                     "//div[@class='MuiTypography-root MuiTypography-body3 mui-style-xckitu' and contains(text(), 'Видеонаблюдение')]").text
            video_surveillance = 1
        except:
            video_surveillance = 0

        video_surveillances.append(video_surveillance)
        print('Video_surveillance:', video_surveillance)

        try:
            parking_space = driver.find_element(By.XPATH,
                                                "//div[@class='MuiTypography-root MuiTypography-body3 mui-style-xckitu' and contains(text(), 'Парковочное место')]").text
            parking_space = 1
        except:
            parking_space = 0

        parking_spaces.append(parking_space)
        print('Parking_space:', parking_space)
        count += 1
        print('-' * 20)

df = pd.DataFrame(
    {
        'Title': flat_titles,
        'Square': squares,
        'Price': prices,
        'Room': rooms,
        'Floor': floors,
        'Building floor': b_floors,
        'Renovation': renovations,
        'Material': materials,
        'Address': addresses,
        'Lift': lifts,
        'Bathroom': bathrooms,
        'Washing machine': wash_machs,
        'TV': tvs,
        'Microwave oven': microwave_ovens,
        'Playground': playgrounds,
        'Furniture': furnitures,
        'Refrigerator': refrigerators,
        'Air conditioner': air_conditioners,
        'Wifi': wifis,
        'Cable': cables,
      'Security': securities,
      'Video surveillance': video_surveillances,
      'Parking space': parking_spaces,
    }
)
df.to_csv('uybor.csv')
