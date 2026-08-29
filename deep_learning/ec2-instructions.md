⏺ The requirements need torch>=2.3. For a T4 GPU (g4dn instance), the right CUDA wheel depends on the driver version
  already on the AMI — that's likely what went wrong before. Here are the corrected steps:

  # 1. Check CUDA driver version first
  nvidia-smi

  # 2. Install conda (if not already on the AMI)
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

  # 3. Create the environment
  conda create -n pred python=3.11 -y
  conda activate pred

  # 4. Install PyTorch — pick ONE line based on nvidia-smi output above:
  #    CUDA 12.x driver  → use cu121
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  #    CUDA 11.x driver  → use cu118
  pip install torch --index-url https://download.pytorch.org/whl/cu118

  # 5. Install remaining dependencies (torch must already be installed to avoid pip
  #    pulling CPU-only torch from PyPI via the requirements file)
  cd mlb
  pip install -r deep_learning/requirements-deep-learning.txt


  # 5. Verify GPU is visible
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

  # 6. Build the feature store (reads from S3, writes locally)
  PYTHONPATH=deep_learning python -m mlb_dl.train build-feature-store \
    --source s3://mlb-265753586044-us-east-1-an/data \
    --output deep_learning/artifacts/feature_store \
    --season-start 2015  # DO NOT use 1950: pre-Statcast rows have no pitch tracking and a 1950-2024 train split voided the 2026-08-28 architecture sweep (see mlb_dl/train_unified.py _STATCAST_MIN_DATE)

  # 7. Train the pregame model (runs eval automatically at end)
  PYTHONPATH=deep_learning python -m mlb_dl.train fit-pregame \
    --feature-store deep_learning/artifacts/feature_store \
    --output deep_learning/artifacts/pregame_cnn \
    --epochs 20

  # 8. Check results
  cat deep_learning/artifacts/pregame_cnn/history.json
  cat deep_learning/artifacts/pregame_cnn/eval_test/evaluation.json

  That's it. Step 6 takes 10–25 min (network-bound reading Parquet from S3), step 7 takes 10–20 min on the A10G. The
  evaluation JSON files are written automatically when training finishes.

  Make sure your EC2 instance's IAM role has s3:GetObject and s3:ListBucket on your bucket — otherwise step 6 will fail
  with access denied.
