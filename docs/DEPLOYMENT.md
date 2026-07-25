# Deployment guide

This guide separates three different goals: a public GitHub repository, a no-install online demo, and a future PyPI package.

## 1. Public GitHub repository

The repository is prepared so large local datasets and archives are ignored automatically.

1. Create a new public repository named `Chorus` on GitHub.
2. Do not add another README, `.gitignore`, or license on GitHub; these files already exist locally.
3. Open GitHub Desktop.
4. Add the local `Chorus` folder as a repository.
5. Commit with the message `Initial public release`.
6. Select **Publish repository** and make sure **Keep this code private** is disabled.
7. Replace the package repository links in `pyproject.toml` only after the final GitHub URL is known.

Before publishing, Git should include the source, README, license, tests, launchers, and docs. It should not include `bench/`, `archive/`, `tessdata/`, `out/`, `.venv/`, or ZIP files.

## 2. Hosted no-install demo

A hosted demo is the easiest experience for recruiters, executives, and non-technical visitors. They click a link, upload a non-confidential image, and see the result without installing Python.

The project includes:

- `app.py` as the hosted-app entry point
- `chorus/web.py` as the browser interface
- `requirements.txt` for the public EasyOCR demo

### Hugging Face Spaces approach

1. Create a free Hugging Face account.
2. Create a new **Space**.
3. Select **Gradio** as the SDK.
4. Choose public visibility.
5. Copy the public repository files into the Space or connect the GitHub repository.
6. Use CPU Basic for an initial demo.
7. Wait for the build to finish, then copy the Space URL into the GitHub README and CV.

The public interface has exactly two modes. Standard Fusion uses EasyOCR, PaddleOCR, and Tesseract. Maximum Performance adds GOT-OCR 2.0. EasyOCR-only operation is not exposed because multi-engine fusion is the core product behavior.

### Public-demo privacy

A public Space should display the included warning. Do not invite users to upload private contracts, IDs, medical records, invoices containing personal information, or company-confidential documents.

## 3. Windows one-click local demo

`START_CHORUS.bat` performs these steps automatically:

1. Finds Python.
2. Creates `.venv` if needed.
3. Ensures that Tesseract is installed.
4. Asks whether GOT-OCR should be installed and explains that it is required for Maximum Performance mode.
5. Installs Standard Fusion or Maximum Performance dependencies.
6. Starts the browser interface.

The only prerequisite is Python 3.10-3.13 and internet access during the first setup.

## 4. Python package and PyPI

The project can already be built as a wheel and installed locally:

```bash
python -m build
pip install dist/chorus_ocr-0.1.0-py3-none-any.whl
```

Publishing to PyPI requires a PyPI account, a unique package name, trusted publishing or an API token, and a release decision. Until that publication happens, do not claim that `pip install chorus-ocr` works from the public package index.

After publishing, users will be able to run:

```bash
pip install "chorus-ocr[demo]"
chorus-demo
```

## Recommended portfolio order

1. Publish the cleaned GitHub repository.
2. Confirm that GitHub Actions passes.
3. Deploy the EasyOCR browser demo.
4. Add the live-demo URL near the top of README.
5. Add the GitHub and demo URLs to the CV.
6. Consider PyPI only after the GitHub and demo experience are stable.
