
PREPROCESSOR_KEY = "preprocessors"
POSTPROCESSORS_KEY = "postprocessors"


def add_preprocessing(base_class):
    base_class_init = base_class.__init__

    def new_init(self, func, data, *args, **kwargs):
        self._preprocessors = kwargs.pop(PREPROCESSOR_KEY, None)
        if self._preprocessors:
            for preprocessor in self._preprocessors:
                data = preprocessor(data)
        base_class_init(self, func, data, *args, **kwargs)

    if hasattr(base_class, "explain_local"):
        if not hasattr(base_class, "_explain_local"):
            base_class._explain_local = base_class.explain_local

        def new_explain_local(self, data, *args, **kwargs):
            import pdb; pdb.set_trace()
            if self._preprocessors:
                for preprocessor in self._preprocessors:
                    data = preprocessor(data)
            import pdb; pdb.set_trace()
            return base_class._explain_local(self, data, *args, **kwargs)
        base_class.explain_local = new_explain_local

    base_class.__init__ = new_init
    return base_class


def add_postprocessing(base_class):
    base_class_init = base_class.__init__

    def new_init(self, func, data, *args, **kwargs):
        self._postprocessors = kwargs.pop(POSTPROCESSORS_KEY, None)
        base_class_init(self, func, data, *args, **kwargs)

    if hasattr(base_class, "explain_local"):
        base_class._original_explain_local = base_class.explain_local

        def new_explain_local(self, *args, **kwargs):
            post_processors = self._postprocessors
            explanation = base_class._original_explain_local(self, *args, **kwargs)
            if post_processors:
                for postprocessor in post_processors:
                    explanation = postprocessor(explanation)
            return explanation
        base_class.explain_local = new_explain_local

    if hasattr(base_class, "explain_global"):
        base_class._original_explain_global = base_class.explain_global

        def new_explain_global(self, *args, **kwargs):
            post_processors = self._postprocessors
            explanation = base_class._original_explain_global(self, *args, **kwargs)
            if post_processors:
                for postprocessor in post_processors:
                    explanation = postprocessor(explanation)
            return explanation
        base_class.explain_global = new_explain_global

    base_class.__init__ = new_init
    return base_class
