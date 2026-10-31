import requests
import pandas as pd

API_KEY = "132094c59b8eb0e7455c84bc5f1e8e8891e099cf"

query_asia = 'area:asia grp:"land mammals"'
query_europe = 'area:europe grp:"land mammals"'
query_america = 'area:america grp:"land mammals"'
query_africa = 'area:africa grp:"land mammals"'
query_australia = 'area:australia grp:"land mammals"'

url = "https://xeno-canto.org/api/3/recordings"

all_recordings = []

page = 1

while True:

    params = {
        "query": query_australia,
        "key": API_KEY,
        "page": page,
        "per_page": 100
    }

    data = requests.get(url, params=params).json()

    all_recordings.extend(data["recordings"])

    print("Downloaded data from page:", page)

    if page >= int(data["numPages"]):
        break

    page += 1


# Convert to CSV
df = pd.DataFrame(all_recordings)

folderPath_Asia = "/Volumes/Maniac/Asia Land Mammels"
folderPath_Europe = "/Volumes/Maniac/Europe Land Mammels"
folderPath_America = "/Volumes/Maniac/America Land Mammels"
folderPath_Africa = "/Volumes/Maniac/Africa Land Mammels"
folderPath_Australia = "/Volumes/Maniac/Australia Land Mammels"

csvFilePath_Asia = f"{folderPath_Asia}/Asia_Land_Mammals.csv"
csvFilePath_Europe = f"{folderPath_Europe}/Europe_Land_Mammals.csv"
csvFilePath_America = f"{folderPath_America}/America_Land_Mammals.csv"
csvFilePath_Africa = f"{folderPath_Africa}/Africa_Land_Mammals.csv"
csvFilePath_Australia = f"{folderPath_Australia}/Australia_Land_Mammals.csv"



df.to_csv(
    csvFilePath_Australia,
    index=False
)

print("\nCSV created successfully!")
print("Total recordings:", len(df))