from pathlib import Path

from flask import Flask


def create_app():
    base_dir = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )
    app.config["SECRET_KEY"] = "dev-secret-key"

    from .routes import tracker_bp

    app.register_blueprint(tracker_bp)
    return app
