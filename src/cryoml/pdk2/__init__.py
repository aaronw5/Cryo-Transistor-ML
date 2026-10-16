"""pdk2: second, simulated parameter-extraction benchmarks.

Backends (same interface: ``DEVICES``/``TAGS``, ``device_by_tag``,
``parse_devices``, ``device_curves``, ``stock_card``/``stock_params``,
``bin_index``, ``series_r_per_side``, ``simulate``, ``simulate_many``):

* ``gf180``: GF180MCU 3.3 V BSIM4 bins on the pinned ngspice-41;
* ``ihp``: IHP SG13G2 1.2 V PSP 103.6 (OSDI, prebuilt macOS OpenVAF-Reloaded)
  on Homebrew ngspice-46, isolated from the ngspice-41 configuration.

``boxes``: seven- and wide-parameter boxes per pdk (cold-start spec layout).
``metric``: the paper RRMS with a 1e-10 A trim floor and target-frozen
inclusion.  ``problem``: ``Pdk2Problem`` (the ``ngspice_fit.DeviceProblem``
interface; backend chosen by the ``gf180_``/``ihp_`` tag prefix) and the
``TrustRegionBackend`` adapter.

Scripts in ``scripts/research/pdk2/``: ``gen_bank.py`` / ``ihp_gen_bank.py``,
``make_truth.py`` / ``ihp_make_truth.py`` (tiers a, b, c), ``floors.py``,
``smoke.py``, ``ihp_smoke.py``.
"""
