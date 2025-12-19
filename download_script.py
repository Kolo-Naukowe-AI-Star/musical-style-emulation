import yt_dlp

URLS = [
    "https://www.youtube.com/watch?v=SL5FFdAvaIA&list=PLiyHrD1Lz34wqDQKVULSB6Z2WL4bIMdEu"
]  # taylor swift's full discography

ydl_opts = {
    "format": "m4a/bestaudio/best",
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    "postprocessors": [
        {  # Extract audio using ffmpeg
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",
        }
    ],
    "outtmpl": "audio/%(playlist_index)s - %(title)s.%(ext)s",
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    error_code = ydl.download(URLS)
