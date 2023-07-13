import spotipy
from spotipy.oauth2 import SpotifyOAuth
import apikeys #file containing api keys; not in version control
from main import get_music_titles

scope = ["user-read-private", "playlist-modify-public"]
auth_manager = SpotifyOAuth(client_id = apikeys.CLIENT_ID, client_secret = apikeys.CLIENT_SECRET, redirect_uri = "http://localhost:3000", scope = scope)
sp = spotipy.Spotify(auth_manager=auth_manager)

user_id = sp.me()["id"]

def search_track_url(query):
    search = sp.search(query, 1, 0, "track")
    return search["tracks"]["items"][0]["external_urls"]["spotify"]

new_playlist = sp.user_playlist_create(user_id, "Fetched playlist #1", True, False, "")

titles = get_music_titles("sample vids/slow_fullscreen_sample_vid.mp4")

tracks = []
for title in titles:
    tracks.append(search_track_url(title))
    
sp.playlist_add_items(new_playlist["id"],tracks)



