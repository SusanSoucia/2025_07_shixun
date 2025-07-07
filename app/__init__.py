from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.secret_key = "susan"
    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app