#!/bin/bash

# --- Original Pre-Start Logic ---
export PYTHONUNBUFFERED=1
source /venv/bin/activate

# --- Custom Setup Logic ---
echo "Starting custom setup for IDM-VTON with DressCode..."

# 1. Install Dependencies
echo "Installing dependencies..."
apt-get update && apt-get install -y wget unzip git git-lfs locales sudo libgl1-mesa-glx libglib2.0-0

# 2. Clone Your Modified Repository  
echo "Cloning your forked IDM-VTON repository (dcgradio branch)..."
git clone -b dcgradio https://github.com/el-mashta/IDM-VTON.git /workspace/IDM-VTON
cd /workspace/IDM-VTON

# Verify branch and DressCode features
echo "Current branch: $(git branch --show-current)"
echo "Checking for DressCode support..."
if grep -q "category.*dropdown" gradio_demo/app.py; then
    echo "✅ DressCode category support detected"
else
    echo "❌ DressCode support not found in app.py"
fi
# 3. Install Python Dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# 4. Download and Place Checkpoints from the Hugging Face Space
echo "Downloading and placing required checkpoints..."
git lfs install
git clone https://huggingface.co/spaces/yisol/IDM-VTON /tmp/hf-space

# Ensure checkpoint directories exist
mkdir -p /workspace/IDM-VTON/ckpt/densepose
mkdir -p /workspace/IDM-VTON/ckpt/humanparsing  
mkdir -p /workspace/IDM-VTON/ckpt/openpose/ckpts

# Move the real checkpoints into your application's ckpt directory
cp -r /tmp/hf-space/ckpt/densepose/* /workspace/IDM-VTON/ckpt/densepose/ 2>/dev/null || echo "DensePose checkpoints not found"
cp -r /tmp/hf-space/ckpt/humanparsing/* /workspace/IDM-VTON/ckpt/humanparsing/ 2>/dev/null || echo "Human parsing checkpoints not found"
cp -r /tmp/hf-space/ckpt/openpose/ckpts/* /workspace/IDM-VTON/ckpt/openpose/ckpts/ 2>/dev/null || echo "OpenPose checkpoints not found"

# Clean up the temporary clone
rm -rf /tmp/hf-space

# 5. Download Core Hugging Face Models using your script
echo "Downloading core diffusion models from Hugging Face..."
cd /workspace/IDM-VTON
python download_models.py

# 6. User and SSH Configuration
echo "Creating and configuring users sharif and stephanie..."
# -- Sharif's Setup --
useradd -m -s /bin/bash sharif && echo 'sharif:dull-remind-modern-below' | chpasswd
mkdir -p /home/sharif/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE8reWxn0Px1daHxku3K1jj1dEu4bCghmoGUx1Y3h4ye sharif@keystone" > /home/sharif/.ssh/authorized_keys
chown -R sharif:sharif /home/sharif/.ssh
chmod 700 /home/sharif/.ssh && chmod 600 /home/sharif/.ssh/authorized_keys

# -- Stephanie's Setup --
useradd -m -s /bin/bash stephanie && echo 'stephanie:entire-step-aim-course' | chpasswd
mkdir -p /home/stephanie/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBKTh9yyDDJmGzdIZOwnB9RZ17SBUvTSOs15pDt/kVsS stephaniekim@Stephanies-MacBook-Pro.local" > /home/stephanie/.ssh/authorized_keys
chown -R stephanie:stephanie /home/stephanie/.ssh
chmod 700 /home/stephanie/.ssh && chmod 600 /home/stephanie/.ssh/authorized_keys

# -- Sudo Configuration --
usermod -aG sudo sharif && usermod -aG sudo stephanie
mkdir -p /etc/sudoers.d
echo 'sharif ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/sharif
echo 'stephanie ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/stephanie
chmod 0440 /etc/sudoers.d/sharif && chmod 0440 /etc/sudoers.d/stephanie
echo "Users configured."

# 7. Prepare DressCode Dataset (This logic remains the same)
DATASET_ZIP="/workspace/data/DressCode_v1.zip"
DATA_DIR="/workspace/data/DressCode"
if [ -f "$DATASET_ZIP" ]; then
    echo "DressCode dataset found. Unzipping..."
    unzip -q -o "$DATASET_ZIP" -d "$DATA_DIR"
else
    echo "WARNING: DressCode dataset ZIP not found. Functionality may be limited." >&2
fi

# 8. Verify Setup
echo "Verifying setup..."
ls -la /workspace/IDM-VTON/ckpt/densepose/
ls -la /workspace/IDM-VTON/ckpt/humanparsing/
ls -la /workspace/IDM-VTON/ckpt/openpose/ckpts/
echo "Custom setup complete."

# --- End of Custom Logic ---

# --- App Launch Logic ---
echo "Starting Gradio App in the background..."
cd /workspace/IDM-VTON
python gradio_demo/app.py > /workspace/output.log 2>&1 &

echo "IDM-VTON server started. Check /workspace/output.log for details."