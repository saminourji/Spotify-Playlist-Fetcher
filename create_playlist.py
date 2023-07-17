import spotipy
from spotipy.oauth2 import SpotifyOAuth
import apikeys #file containing api keys; not in version control
from main import get_music_titles

print("Setting up Spotify API")
scope = ["user-read-private", "playlist-modify-public"]
auth_manager = SpotifyOAuth(client_id = apikeys.CLIENT_ID, client_secret = apikeys.CLIENT_SECRET, redirect_uri = "http://localhost:3000", scope = scope)
sp = spotipy.Spotify(auth_manager=auth_manager)

user_id = sp.me()["id"]

def search_track_url(query):
    search = sp.search(query, 1, 1, "track")
    return search["tracks"]["items"][0]["external_urls"]["spotify"]

new_playlist = sp.user_playlist_create(user_id, "Fetched playlist #3", True, False, "")

titles = get_music_titles("sample vids/fast_full_screen_sample_vid.mp4")

links = []
print("Fetching song links")
pos = 0
for title in titles:
    print(title)
    link = search_track_url(title)
    if link not in links:
        links.append(link)
        sp.playlist_add_items(new_playlist["id"], [link], pos)
        pos += 1

print("Success!, Here is the link to your generated playlist: ", new_playlist["external_urls"]["spotify"])



