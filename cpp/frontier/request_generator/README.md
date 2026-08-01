# C++ workload generator

The C++ simulator consumes a normalized workload CSV and does not generate
requests inside the DES runtime. `generate_workload.py` is a standalone Python
producer for that contract; it does not import the Python Frontier simulator.

```powershell
python cpp/frontier/request_generator/generate_workload.py `
  --config cpp/frontier/request_generator/example_workload.json `
  --output outputs/workload.csv
```

Length distributions:

- `fixed`
- `uniform` total tokens with a Prefill:Decode ratio
- bounded `zipf` total tokens with a Prefill:Decode ratio
- independent bounded `bounded_lognormal` Prefill and Decode lengths

Interval distributions:

- `static`
- `poisson` (optional `max_interval_factor`, such as `3` for the historical
  Python Frontier truncation)
- `gamma` parameterized by QPS and coefficient of variation

The default first request arrives at time zero. Set
`first_arrival_at_zero: false` to apply the first sampled interval before the
first request. The output always uses the five-column canonical C++ schema,
including empty session columns for single-turn workloads.
