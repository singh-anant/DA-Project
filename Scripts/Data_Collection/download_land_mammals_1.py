import requests
import os

API_KEY = "132094c59b8eb0e7455c84bc5f1e8e8891e099cf"

query_asia = 'area:asia grp:"land mammals"'
query_europe = 'area:europe grp:"land mammals"'
query_america = 'area:america grp:"land mammals"'
query_africa = 'area:africa grp:"land mammals"' 
query_australia = 'area:australia grp:"land mammals"'


url = "https://xeno-canto.org/api/3/recordings"

folder_Asia = "/Volumes/Maniac/Asia Land Mammels"
folder_Europe = "/Volumes/Maniac/Europe Land Mammels"
folder_America = "/Volumes/Maniac/America Land Mammels"
folder_Africa = "/Volumes/Maniac/Africa Land Mammels"
folder_Australia = "/Volumes/Maniac/Australia Land Mammels"

os.makedirs(folder_Asia, exist_ok=True)
os.makedirs(folder_Europe, exist_ok=True)
os.makedirs(folder_America, exist_ok=True)
os.makedirs(folder_Africa, exist_ok=True)
os.makedirs(folder_Australia, exist_ok=True)

page = 1

while True:

    params = {
        "query": query_america,
        "key": API_KEY,
        "page": page
    }

    data = requests.get(url, params=params).json()

    for recording in data["recordings"]:

        recording_id = recording["id"]
        audio_url = recording["file"]

        extension = os.path.splitext(
            audio_url.split("?")[0]
        )[1]

        if not extension:
            extension = ".mp3"

        filename = f"XC{recording_id}{extension}"
        filepath = os.path.join(folder_America, filename)

        # Skip if already downloaded
        if os.path.exists(filepath):
            continue

        print("Downloading:", filename)

        audio = requests.get(audio_url)

        with open(filepath, "wb") as f:
            f.write(audio.content)

    print("Finished page:", page)

    if page >= int(data["numPages"]):
        break

    page += 1

print("All downloads completed!")