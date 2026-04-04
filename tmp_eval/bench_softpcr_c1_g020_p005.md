Dataset: tmp_eval/job_salary_prediction_dataset.csv (nrows=8000)
seed=42  train_fraction=0.8

Forced prediction (classification):
Target: remote_work
Traceback (most recent call last):
  File "/home/chizoalban2003/Mycelium/scripts/benchmark_job_salary_tasks.py", line 709, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/home/chizoalban2003/Mycelium/scripts/benchmark_job_salary_tasks.py", line 671, in main
    cls_rows = _bench_classification(df, target_col=str(args.cls_target), seed=int(args.seed), train_fraction=float(args.train_fraction))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chizoalban2003/Mycelium/scripts/benchmark_job_salary_tasks.py", line 224, in _bench_classification
    _run_mycelium(
  File "/home/chizoalban2003/Mycelium/scripts/benchmark_job_salary_tasks.py", line 169, in _run_mycelium
    pred = run_physics_prediction(
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chizoalban2003/Mycelium/mycelium_app/physics_predictor.py", line 2481, in run_physics_prediction
    x_raw = x_cat.map(rates).fillna(0.0).to_numpy(dtype="float64")
            ^^^^^^^^^^^^^^^^
  File "/home/chizoalban2003/Mycelium/.venv/lib/python3.11/site-packages/pandas/core/series.py", line 4719, in map
    new_values = self._map_values(arg, na_action=na_action)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chizoalban2003/Mycelium/.venv/lib/python3.11/site-packages/pandas/core/base.py", line 923, in _map_values
    return arr.map(mapper, na_action=na_action)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chizoalban2003/Mycelium/.venv/lib/python3.11/site-packages/pandas/core/arrays/base.py", line 2322, in map
    return map_array(self, mapper, na_action=na_action)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chizoalban2003/Mycelium/.venv/lib/python3.11/site-packages/pandas/core/algorithms.py", line 1732, in map_array
    indexer = mapper.index.get_indexer(arr)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chizoalban2003/Mycelium/.venv/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3955, in get_indexer
    target = target.astype(dtype, copy=False)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chizoalban2003/Mycelium/.venv/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 1052, in astype
    def astype(self, dtype, copy: bool = True):

KeyboardInterrupt
