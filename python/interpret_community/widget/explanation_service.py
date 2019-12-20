from gevent.pywsgi import WSGIServer
from flask import Flask, render_template, url_for, request
from IPython.display import display, HTML
import threading
import jinja2
import os
import json
from .explanation_dashboard_input import ExplanationDashboardInput

class ExplanationDashboard:
    __service = None
    explanations = {}
    model_count = 0
    
    class DashboardService:
        app = Flask(__name__)
        
        app.config['TESTING'] = True
        app.config["EXPLAIN_TEMPLATE_LOADING"] = True

        @app.route('/')
        def hello():
            return "HelloWorld"

        @app.route('/<id>')
        def explanation_visual(id):
            if id in LocalExplanationDash.explanations:
                return render_template( 'dashboard.html', explanation=json.dumps(LocalExplanationDash.explanations[id].dashboard_input), main_js='http://127.0.0.1:5000/static/index.js', app_id='app_123')
            else:
                return "unknown..."
        
        @app.route('/<id>/predict', methods=['POST'])
        def predict(id):
            data = request.get_json(force=True)
            if id in LocalExplanationDash.explanations:
                return LocalExplanationDash.explanations[id].on_predict(data)

    def __init__(self, explanationObject, model=None, *, datasetX=None, trueY=None, classes=None, features=None):
        if not LocalExplanationDash.__service:
            LocalExplanationDash.__service = LocalExplanationDash.DashboardService()
            self._thread = threading.Thread(target=LocalExplanationDash.__service.app.run, daemon=True)
            self._thread.start()
        LocalExplanationDash.model_count += 1
        predict_url = "http://127.0.0.1:5000/{0}/predict".format(str(LocalExplanationDash.model_count))
        LocalExplanationDash.explanations[str(LocalExplanationDash.model_count)] = ExplanationDashboardInput(explanationObject, model, datasetX, trueY, classes, features, predict_url)

        html = "<iframe src='http://127.0.0.1:5000/{0}' width='100%' height='800px' frameBorder='0'></iframe>".format(LocalExplanationDash.model_count)
        if "DATABRICKS_RUNTIME_VERSION" in os.environ:
            _render_databricks(html)
        else: 
            display(HTML(html))
    
# NOTE: Code mostly derived from Plotly's databricks render as linked below:
# https://github.com/plotly/plotly.py/blob/01a78d3fdac14848affcd33ddc4f9ec72d475232/packages/python/plotly/plotly/io/_base_renderers.py
def _render_databricks(html):  # pragma: no cover
    import inspect

    if _render_databricks.displayHTML is None:
        found = False
        for frame in inspect.getouterframes(inspect.currentframe()):
            global_names = set(frame.frame.f_globals)
            target_names = {"displayHTML", "display", "spark"}
            if target_names.issubset(global_names):
                _render_databricks.displayHTML = frame.frame.f_globals["displayHTML"]
                found = True
                break

        if not found:
            msg = "Could not find DataBrick's displayHTML function"
            log.error(msg)
            raise RuntimeError(msg)

    _render_databricks.displayHTML(html)


_render_databricks.displayHTML = None
    

        
        