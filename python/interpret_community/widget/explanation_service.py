from gevent.pywsgi import WSGIServer
from flask import Flask, render_template, url_for
from IPython.display import display, HTML
import threading
import jinja2
import os

class LocalExplanationDash:
    __service = None
    explanations = {}
    env = jinja2.Environment(loader=jinja2.PackageLoader(__name__, 'templates'))
    default_template = env.get_template("dashboard.html")
    
    class DashboardService:
        app = Flask(__name__)
        
        app.config['TESTING'] = True
        app.config["EXPLAIN_TEMPLATE_LOADING"] = True
        def __init__(self, base_url=None):
            self.config = {}
            self.call_count = 0

        def register(self, id, explanation):
            LocalExplanationDash.explanations[str(id)] = explanation

        @app.route('/')
        def hello():
            return "HelloWorld"

        @app.route('/<id>')
        def explanation_visual(id):
            if id in LocalExplanationDash.explanations:
                return LocalExplanationDash.default_template.render(explanation=LocalExplanationDash.explanations[id], main_js='http://127.0.0.1:5000/static/index.js', app_id='app_123')

            return "unknown"

    def __init__(self, explanationObject, model=None):
        print(__name__)
        if not LocalExplanationDash.__service:
            LocalExplanationDash.__service = LocalExplanationDash.DashboardService()
            self._thread = threading.Thread(target=LocalExplanationDash.__service.app.run, daemon=True)
            self._thread.start()
        LocalExplanationDash.__service.call_count += 1

        LocalExplanationDash.__service.register(LocalExplanationDash.__service.call_count, explanationObject)

        html = "<iframe src='http://127.0.0.1:5000/{0}' width='100%' height='800px' frameBorder='0'></iframe>".format(LocalExplanationDash.__service.call_count)
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
    

        
        