import os

import yt_dlp


def download_playlist_as_audio(playlist_url, output_folder="data"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "outtmpl": f"{output_folder}/%(title)s.%(ext)s",
        "noplaylist": False,
        "quiet": False,
    }

    print(f"Starting download to: {os.path.abspath(output_folder)}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([playlist_url])
            print("\nDownload Complete!")
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    URL = "https://www.youtube.com/playlist?list=PL-xy0dPHNOspoP2en2JExAXqQMy3g5_rL"
    download_playlist_as_audio(URL)
