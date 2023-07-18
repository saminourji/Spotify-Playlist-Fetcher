import spotipy
from spotipy.oauth2 import SpotifyOAuth
import apikeys #file containing api keys; not in version control
from main import get_music_titles

print("Setting up Spotify API")
scope = ["user-read-private", "playlist-modify-public"]
auth_manager = SpotifyOAuth(client_id = apikeys.CLIENT_ID, client_secret = apikeys.CLIENT_SECRET, redirect_uri = "http://localhost:3000", scope = scope)
sp = spotipy.Spotify(auth_manager=auth_manager)

user_id = sp.me()["id"]

# only get top result
def search_track_url(query):
    search = sp.search(query, 1, 1, "track")
    return search["tracks"]["items"][0]["external_urls"]["spotify"]

# get best of the top n results
def search_best_url(tuple, n):
    links = []
    titles = []
    if len(tuple) == 2:
        search = sp.search(tuple[0].join(" ").join(tuple[1]), n, 0, "track")
        for i in range(n):
            links.append(search["tracks"]["items"][i]["external_urls"]["spotify"])
            titles.append(search["tracks"]["items"][i]["name"])

new_playlist = sp.user_playlist_create(user_id, "Fetched playlist #4", True, False, "")

song_info = get_music_titles("sample  vids/slow_fullscreen_sample_vid.mp4")

links = []
print("Fetching song links")
pos = 0
for info in song_info:
    clean_title = [x for x in info.split("\n") if x != ""]
    print(len(clean_title), clean_title)
    link = search_best_url(info, 3)
    if link not in links:
        links.append(link)
        sp.playlist_add_items(new_playlist["id"], [link], pos)
        pos += 1

print("Success!, Here is the link to your generated playlist: ", new_playlist["external_urls"]["spotify"])



