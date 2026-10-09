# Running tests

See the [testing guide](../docs/development/testing.md) for environment setup,
CPU and CUDA commands, and simulated disk-to-disk examples.
[Test review rules](CONTRIBUTING.md) define permitted test categories and physics
acceptance requirements; the documentation site includes that same file.

From an installed development environment:

```powershell
uv run --no-sync python -m pytest tests/cpu --strict-markers -q
```
