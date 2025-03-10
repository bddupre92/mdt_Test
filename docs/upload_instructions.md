# Upload to TestPyPI Instructions

Since we're having issues with the interactive upload, here are instructions to upload using your TestPyPI API token directly:

1. Make sure you're registered on TestPyPI: https://test.pypi.org/account/register/

2. Create an API token if you haven't already:
   - Go to https://test.pypi.org/manage/account/
   - Click on "API tokens"
   - Create a new token with "Upload" scope

3. Run the following command, replacing YOUR_TOKEN with your actual TestPyPI token:

```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD=YOUR_TOKEN python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

4. After successful upload, you can install your package with:

```bash
pip install --index-url https://test.pypi.org/simple/ meta_optimizer_mdt_test
```

## Troubleshooting 403 Forbidden Errors

If you're still getting 403 Forbidden errors, try these steps:

1. Verify your token has "Upload" permissions in your TestPyPI account settings
2. Make sure your account is verified (email verification)
3. Check if another package with the same name exists (though we've tried to make the name unique)
4. Try a completely different name like `meta_optimizer_bddupre92`

## Alternative: Local Testing

If TestPyPI is giving you trouble, you can test locally:

```bash
# Install in development mode
pip install -e .

# Test the package is working
python -c "import meta_optimizer; print(meta_optimizer.__version__)"
```

This will allow you to test your package functionality without uploading to TestPyPI.

## Uploading to the Main PyPI Repository

Once you've tested your package on TestPyPI and confirmed it works correctly, you can upload it to the main PyPI repository:

```bash
# Clean build directories
rm -rf dist build meta_optimizer_mdt_test.egg-info

# Build the package
python -m build

# Upload to PyPI (you'll be prompted for username and password)
python -m twine upload dist/*
```

Alternatively, you can use the provided script:

```bash
./upload_to_pypi.sh
```

After uploading to PyPI, your package will be available for anyone to install using:

```bash
pip install meta_optimizer_mdt_test
```

### API Token for PyPI

For secure uploads, we recommend using an API token:

1. Go to https://pypi.org/manage/account/
2. Create an API token with upload permissions
3. Use the token as your password when prompted by twine (with your username)

### Avoiding Re-upload Errors

PyPI does not allow re-uploading files with the same version number. If you need to make changes:

1. Increment the version number in:
   - pyproject.toml
   - setup.py
   - meta_optimizer/__init__.py
2. Rebuild and upload the package
