# Release Process for rag-bridge-kit

## 1. Update version

Edit `pyproject.toml` and `src/rag_bridge_kit/__init__.py`:

```
version = "0.2.0"
__version__ = "0.2.0"
```

## 2. Run tests

```bash
pip install -e ".[dev]"
python -m pytest -v
```

## 3. Build

```bash
python -m build
```

## 4. Test on TestPyPI (optional)

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ rag-bridge-kit
```

## 5. Publish to PyPI

```bash
twine upload dist/*
```

Or create a GitHub Release and the `publish.yml` workflow will handle it automatically.

## 6. Tag the release

```bash
git tag v0.2.0
git push origin v0.2.0
```
