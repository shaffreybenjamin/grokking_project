## Setup

1. Clone the repo and `cd` into it.
2. Create the conda environment:

       conda env create -f environment.yml
       conda activate mathematics_thesis

3. Run the install script:

       bash install.sh

   The script auto-detects whether the machine has an NVIDIA GPU and
   installs the appropriate PyTorch build (CUDA or CPU-only), then
   installs all remaining dependencies from `requirements.txt`.

4. Confirm setup succeeded — the script's final output should show
   `CUDA available: True` on GPU machines and `False` on CPU-only ones.

### Adding new packages

After installing a package, add it to `requirements.txt` and commit:

    pip install <package>
    # then add <package> to requirements.txt and:
    git add requirements.txt
    git commit -m "Add <package>"
    git push

Do not add `torch` or `torchvision` to `requirements.txt` — they're
handled by `install.sh`.