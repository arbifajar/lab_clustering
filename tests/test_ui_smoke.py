"""Smoke test alur sintaks antarmuka tanpa menjalankan server Streamlit.

Bukan pengganti browser E2E: library Streamlit tidak tersedia pada lingkungan CI offline.
"""
import runpy
import sys
import types
from pathlib import Path

from src.benchmark import run_benchmark
from src.data import load_baskets


class FakeSession(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__('streamlit')
        self.sidebar = self
        self.session_state = FakeSession()
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def cache_data(self, *args, **kwargs): return lambda fn: fn
    def columns(self, count, **kwargs):
        return [self] * (count if isinstance(count, int) else len(count))
    def tabs(self, labels): return [self] * len(labels)
    def radio(self, label, options, **kwargs): return options[0]
    def selectbox(self, label, options, **kwargs): return options[kwargs.get('index', 0)]
    def slider(self, label, lo, hi, value, *args, **kwargs): return value
    def number_input(self, label, *args, **kwargs): return kwargs.get('value', args[2] if len(args) > 2 else (args[0] if args else 0))
    def checkbox(self, label, *args, **kwargs): return kwargs.get('value', False)
    def multiselect(self, label, options, **kwargs): return kwargs.get('default', [])
    def button(self, label, **kwargs): return False
    def __getattr__(self, key):
        return lambda *args, **kwargs: self


def test_app_first_render_and_existing_benchmark(monkeypatch):
    st = FakeStreamlit()
    monkeypatch.setitem(sys.modules, 'streamlit', st)
    project = Path(__file__).resolve().parents[1]
    runpy.run_path(str(project / 'app.py'), run_name='__main__')
    assert st.session_state.run is None
    _, baskets = load_baskets()
    result = run_benchmark(tuple(baskets[:200]), fractions=(25,), supports=(3,), repeats=1, measure_memory=False)
    st.session_state.run = result
    st.session_state.run_signature = ((25,), (3,), 3, 42, 3, True)
    runpy.run_path(str(project / 'app.py'), run_name='__main__')
