import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
import matplotlib.pyplot as plt
import re
import numpy as np
import scipy.stats as stats
import seaborn as sns
import pandas as pd
url = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/malopolskie/krakow/krakow/krakow?limit=36&ownerTypeSingleSelect=ALL&by=DEFAULT&direction=DESC&viewType=listing&page="


headers = {
    "User-Agent": "Mozilla/5.0"
}

neighborhoods = {
    'Grzegórzki', 'Swoszowice', 'Stare Miasto', 'Prądnik Czerwony', 'Zwierzyniec', 'Bronowice', 'Prądnik Biały', 'Dębniki', 'Krowodrza', 'Łagiewniki-Borek Fałęcki', 'Podgórze Duchackie', 'Bieżanów-Prokocim', 'Podgórze', 'Czyżyny', 'Mistrzejowice', 'Bieńczyce', 'Wzgórza Krzesławickie', 'Nowa Huta'
}

pages = []

for page in range(1, 32):
    ready_url = f"{url}{page}"
    response = requests.get(ready_url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        pages.append(soup)

        print(f"Status: {response.status_code}. Przetwarzam strone: {page}")


flat_info = {}

for page_index, page in enumerate(pages):
    all_flats = page.find('div', attrs={'data-cy': 'search.listing.organic'}).find_all('li')
    #print(all_flats) #all_flats dziala
    for flat_index, flat in enumerate(all_flats):
        unique_key = f"{page_index}_{flat_index}"
        
        title = flat.find('div', class_="css-12h460e e1nxvqrh1")
        
        if title:
            title = title.text.strip()
            
            district_match = re.search(r', ([^,]+), Kraków', title)
            district = district_match.group(1).strip() if district_match else None
        price = flat.find('span', class_="css-1uwck7i evk7nst0")
        
        if price:
            price = price.text.strip()
            print(price) #dziala
        
        information_of_flat = flat.find('dl', class_="css-uki0wd e1clni9t1")
        
        if information_of_flat:
            information_of_flat = information_of_flat.text.strip()
            print(information_of_flat)
            num_of_rooms_match = re.search(r'Liczba pokoi\s*(\d+)', information_of_flat)
            num_of_rooms = num_of_rooms_match.group(1) if num_of_rooms_match else None
            
            perimeter_match = re.search(r'Powierzchnia\s*([\d,]+)\s*m²', information_of_flat)
            perimeter = perimeter_match.group(1).replace(',', '.') if perimeter_match else None
            
            price_per_meter_squared_match = re.search(r'Cena za metr kwadratowy\s*([\d\s]+)zł/m²', information_of_flat)
            price_per_meter_squared = price_per_meter_squared_match.group(1).replace('\xa0', '').replace(' ', '') if price_per_meter_squared_match else None
            
            url = 'https://www.otodom.pl' + flat.find('a', attrs={'data-cy': 'listing-item-link'})['href']
            
            flat_info[unique_key] = {
                'title': title,
                'price': price,
                'num_of_rooms': num_of_rooms,
                'perimeter': perimeter,
                'price_per_meter_squared': price_per_meter_squared,
                'url': url,
                'district': district
                
            }


for unique_key, info in flat_info.items():
    print(f"Unique Key: {unique_key}")
    print(f"Title: {info['title']}")
    print(f"Price: {info['price']}")
    print(f"Number of Rooms: {info['num_of_rooms']}")
    print(f"Perimeter: {info['perimeter']}")
    print(f"Price per Meter Squared: {info['price_per_meter_squared']}")
    print(f"URL: {info['url']}")
    print(f"District: {info['district']}\n")
    

# Konwersja danych na listę list dla tabulate
table = []
for unique_key, info in flat_info.items():
    row = [unique_key]
    row.extend([info[key] for key in ['title', 'price', 'num_of_rooms', 'perimeter', 'price_per_meter_squared', 'url', 'district']])
    table.append(row)

# Wyświetlanie danych w postaci tabeli
headers = ['Unique Key', 'Title', 'Price', 'Number of Rooms', 'Perimeter', 'Price per Meter Squared', 'URL', 'District']
#print(tabulate(table, headers=headers, tablefmt="pipe"))

print("\nNajdrozsze mieszkanie:\t")
filtered_flats = {k: v for k, v in flat_info.items() if v['price'] != "Zapytaj o cenę"}

# Find and display the most expensive flat
if filtered_flats:
    most_expensive_flat = max(filtered_flats.items(), key=lambda x: x[1]['price'])

    print("\nNajdroższe mieszkanie:\t")
    print(f"Unique Key: {most_expensive_flat[0]}")
    print(f"Title: {most_expensive_flat[1]['title']}")
    print(f"Price: {most_expensive_flat[1]['price']}")
    print(f"Number of Rooms: {most_expensive_flat[1]['num_of_rooms']}")
    print(f"Perimeter: {most_expensive_flat[1]['perimeter']}")
    print(f"Price per Meter Squared: {most_expensive_flat[1]['price_per_meter_squared']}")
    print(f"URL: {most_expensive_flat[1]['url']}")
    print(f"District: {most_expensive_flat[1]['district']}")

district_prices = {}
for info in filtered_flats.values():
    district = info['district']
    if district and info['price_per_meter_squared']:
        price_per_meter_squared = float(info['price_per_meter_squared'])
        if district in district_prices:
            district_prices[district].append(price_per_meter_squared)
        else:
            district_prices[district] = [price_per_meter_squared]

average_prices = {district: sum(prices)/len(prices) for district, prices in district_prices.items()}

sorted_average_prices = sorted(average_prices.items(), key=lambda x: x[1])

# Prepare data for plotting
sorted_districts = [item[0] for item in sorted_average_prices]
sorted_prices = [item[1] for item in sorted_average_prices]

# Plot the data
plt.figure(figsize=(12, 8))
plt.bar(sorted_districts, sorted_prices, color='skyblue')
plt.xlabel('Districts')
plt.ylabel('Average Price per Meter Squared (PLN)')
plt.title('Average Price per Meter Squared by District in Krakow')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Calculate average price per meter squared by number of rooms
rooms_prices = {}
for info in filtered_flats.values():
    num_of_rooms = info['num_of_rooms']
    if num_of_rooms and info['price_per_meter_squared']:
        price_per_meter_squared = float(info['price_per_meter_squared'])
        if num_of_rooms in rooms_prices:
            rooms_prices[num_of_rooms].append(price_per_meter_squared)
        else:
            rooms_prices[num_of_rooms] = [price_per_meter_squared]

average_prices_by_rooms = {rooms: sum(prices)/len(prices) for rooms, prices in rooms_prices.items()}

# Sort by number of rooms
sorted_average_prices_by_rooms = sorted(average_prices_by_rooms.items(), key=lambda x: int(x[0]))

# Prepare data for plotting
sorted_rooms = [item[0] for item in sorted_average_prices_by_rooms]
sorted_prices_by_rooms = [item[1] for item in sorted_average_prices_by_rooms]

# Plot the data
plt.figure(figsize=(10, 6))
plt.bar(sorted_rooms, sorted_prices_by_rooms, color='skyblue')
plt.xlabel('Number of Rooms')
plt.ylabel('Average Price per Meter Squared (PLN)')
plt.title('Average Price per Meter Squared by Number of Rooms')
plt.tight_layout()
plt.show()


# Number of flats per district
district_counts = {}
for info in filtered_flats.values():
    district = info['district']
    if district:
        if district in district_counts:
            district_counts[district] += 1
        else:
            district_counts[district] = 1

# Sort districts by number of flats
sorted_district_counts = sorted(district_counts.items(), key=lambda x: x[1])

# Prepare data for plotting
sorted_districts_by_count = [item[0] for item in sorted_district_counts]
sorted_counts = [item[1] for item in sorted_district_counts]

# Plot the data
plt.figure(figsize=(12, 8))
plt.bar(sorted_districts_by_count, sorted_counts, color='skyblue')
plt.xlabel('Districts')
plt.ylabel('Number of Flats')
plt.title('Number of Flats Listed per District in Krakow')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Prices per meter squared for normal distribution plot
prices_per_meter = [float(info['price_per_meter_squared']) for info in filtered_flats.values() if info['price_per_meter_squared']]

# Fit a normal distribution
mu, std = stats.norm.fit(prices_per_meter)

# Plot the histogram and the fitted normal distribution
plt.figure(figsize=(10, 6))
plt.hist(prices_per_meter, bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black')

# Plot the normal distribution curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

plt.xlabel('Price per Meter Squared (PLN)')
plt.ylabel('Density')
plt.title(f'Fit of Normal Distribution: mu = {mu:.2f}, std = {std:.2f}')
plt.tight_layout()
plt.show()

# Prepare data for boxplot
district_price_data = []
district_labels = []

for district, prices in district_prices.items():
    district_price_data.append(prices)
    district_labels.append(district)

# Create the boxplot
plt.figure(figsize=(12, 8))
plt.boxplot(district_price_data, labels=district_labels, patch_artist=True, vert=False)
plt.xlabel('Price per Meter Squared (PLN)')
plt.title('Boxplot of Prices per Meter Squared by District')
plt.tight_layout()
plt.show()

# Convert to DataFrame for heatmap
district_price_df = pd.DataFrame.from_dict(average_prices, orient='index', columns=['Average Price per Meter Squared'])

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(district_price_df.sort_values(by='Average Price per Meter Squared'), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Average Prices per Meter Squared by District')
plt.xlabel('Districts')
plt.ylabel('Average Price per Meter Squared (PLN)')
plt.tight_layout()
plt.show()

# Convert data to long-form DataFrame for Seaborn
price_data = []
for district, prices in district_prices.items():
    for price in prices:
        price_data.append({'District': district, 'Price per Meter Squared': price})

price_df = pd.DataFrame(price_data)

# Create the violin plot
plt.figure(figsize=(12, 8))
sns.violinplot(x='Price per Meter Squared', y='District', data=price_df, scale='width', palette='coolwarm')
plt.title('Violin Plot of Prices per Meter Squared by District')
plt.xlabel('Price per Meter Squared (PLN)')
plt.ylabel('Districts')
plt.tight_layout()
plt.show()

