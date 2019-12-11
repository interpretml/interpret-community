from gevent.pywsgi import WSGIServer
from flask import Flask

class LocalExplanationDash:
    __service = None
    class __DashboardService:
        def __init__(self):
            app = Flask(__name__)
            app.run()
        @app.route("/")
        def base():
            return "Hello world"
    
    def __init__(self, id, val):
        if not LocalExplanationDash.__service:
            LocalExplanationDash.__service = LocalExplanationDash.__DashboardService()
    
        
        