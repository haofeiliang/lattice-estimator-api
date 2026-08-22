"""Lattice estimator HTTP application.

Code map for new contributors:

* :mod:`src.app` is the ASGI/HTTP entry point and defines every public route.
* :mod:`src.process` starts and supervises one isolated Sage process per request.
* :mod:`src.worker` is the JSON stdin/stdout entry point inside that process.
* :mod:`src.adapter` converts public models to calls into ``lattice-estimator``
  and normalizes the upstream results.
* :mod:`src.models` defines the request and response protocol shared by them.
"""
