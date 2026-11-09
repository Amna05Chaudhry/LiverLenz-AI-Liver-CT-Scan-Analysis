"""Run the web app: ``python -m app`` then open http://127.0.0.1:5000"""
import os

from . import create_app

if __name__ == "__main__":
    app = create_app()
    print("Loading segmentation model...")
    _ = app.extensions["segmenter"].model  # load once at startup, not on the first request
    port = int(os.getenv("PORT", "5000"))
    print(f"LiverLenz AI running at http://127.0.0.1:{port}")
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=port, debug=False)
