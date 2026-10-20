# rustler

Rust context sampler exposed to Python as `rustler.Sampler`.

- `src/fly.rs`: temporal context sampling, target masking, and typed FK edges.
- `src/common.rs`: serialized graph types matching the frozen data artifacts.
- `src/lib.rs`: Python extension entry point.

The sampler reads the artifacts installed by `data/download.py` and honors
`NANORT_PRE_ROOT`. Build it through `./run.sh`, or after `pixi install --locked`:

```bash
cd rustler
pixi run maturin develop --uv --release
```
