"""Run the web app: ``python -m app`` then open http://127.0.0.1:5000"""
import os

from . import create_app

if __name__ == "__main__":
    app = create_app()
    m = app.extensions["cascade"]
    if m.available:
        print("Loading segmentation models...")
        m._sessions()
    else:
        print("WARNING: models/liver_stage1.onnx and models/tumour_stage2.onnx not found; see README.")
    port = int(os.getenv("PORT", "5000"))
    print(f"LiverLenz AI running at http://127.0.0.1:{port}")
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=port, debug=False)
