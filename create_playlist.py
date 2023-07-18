import spotipy
from spotipy.oauth2 import SpotifyOAuth
import apikeys #file containing api keys; not in version control
from main import get_music_titles
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

print("Setting up Spotify API")
scope = ["user-read-private", "playlist-modify-public"]
auth_manager = SpotifyOAuth(client_id = apikeys.CLIENT_ID, client_secret = apikeys.CLIENT_SECRET, redirect_uri = "http://localhost:3000", scope = scope)
sp = spotipy.Spotify(auth_manager=auth_manager)
path = "sample vids/slow_fullscreen_sample_vid.mp4"

user_id = sp.me()["id"]

# only get top result
def search_track_url(query):
    search = sp.search(query, 1, 1, "track")
    return search["tracks"]["items"][0]["external_urls"]["spotify"]

# get best of the top n results
def search_best_url(tuple, n):
    assert(len(tuple) == 2)

    # (title ratio, link, result nb)
    best = None
    
    query = " - " + tuple[0] + " " + tuple[1].split(",")[0]
    search = sp.search(query, n, 0, "track")
    # print(query)
    for i in range(n):
        compare_title = similar(search["tracks"]["items"][i]["name"], tuple[0])
        # print(f'TITLE score: {compare_title} between actual: {tuple[0]} and fetched: {search["tracks"]["items"][i]["name"]}')
        compare_artist = similar(search["tracks"]["items"][i]["artists"][0]["name"], tuple[1].split(",")[0])
        # print(f'ARTIST score: {compare_artist} between actual: {tuple[1].split(",")[0]} and fetched: {search["tracks"]["items"][i]["artists"][0]["name"]}')
        if best == None or (compare_title > best[0] and compare_artist > 0.2):
            best = (compare_title, search["tracks"]["items"][i]["external_urls"]["spotify"], i)
            #if perfect match, no need to look at the other cases
            if compare_title == 1:
                return best
    # print(f"{best}for {tuple[0]} by {tuple[1]}")
    return best

new_playlist = sp.user_playlist_create(user_id, f"Playlist for {path}", True, False, "")

song_info = get_music_titles(path, 3)
# song_info = ['Beginnings (feat. East Forest)\nMC YOGI, East Forest\n', 'Yoste\n', 'The Healing\n\nEssie Jain\n', 'Gold Flow\n\nOJ Taz Rashid\n', 'Limitations\n\nEast Forest\n', 'Inspiration Drive\nDJ Taz Rashid\n', 'Chillaxing\n\nSol Rising\n', 'I Am - Krishan Liquid Mix\n\n. Nirinjan Kaur, Matthew Schoening, Ram D\n', 'Vega\n\nillo\n', 'Opening\n\nEssie Jain\n', 'Floating Sweetness\n\nDJ Drez, Domonic Dean Breaux\n', 'Kaonashi\n\nYoste\n', 'I Will Love You\nSol Rising\n', 'Waves\nIHF, Cyn\n', 'Wellspring - DJ Taz Rashid, Mo\n\nScott Nice, Nathan Hall, DJ Taz Rashid, Mor\n', 'Closing Meditation\n\nShantala\n', 'By Your Grace/Jai Gurudev\n\nKrishna Das\n', 'One River\nBenjy Wertheimer, John De Kadt\n', 'Shanti (Peace Out)\nMC YOGI\n', 'Angels Prayer\n\n»s Hoskins, Cat McCarthy, Manorama, Janak\n', 'In Dreams\nJai-Jagdeesh\n', 'Light Me Up\n\nDJ Drez, Marti Nikko\n', 'Heart Chakra\n\nBeautiful Chorus\n', 'Eyelids Gently Dreaming\n\nSteve Hauschildt\n', 'Angels Prayer\n\n-s Hoskins, Cat McCarthy, Manorama, Janak\n', 'Nectar Drop\nDJ Drez\n', 'Introspection\n\nLaraaji\n']

omitted = 0
links = []
print("Fetching song links")
for info in song_info:
    split_info = [x for x in info.split("\n") if x != ""] #expected to put in format (title, artist) 
    if len(split_info) == 2:
        ratio, link, _ = search_best_url(split_info, 3)
        if link not in links:
            if ratio >= 0.5:
                links.append(link)
            else: omitted += 1
    else: omitted += 1
sp.playlist_add_items(new_playlist["id"], links, 0)
sp.playlist_change_details(new_playlist["id"], description = f"Omitted song due to analysis error: {omitted}")
print(omitted)
print("Success!, Here is the link to your generated playlist: ", new_playlist["external_urls"]["spotify"])



