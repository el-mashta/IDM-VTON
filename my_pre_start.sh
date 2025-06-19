#!/bin/bash

# --- Original Pre-Start Logic ---
export PYTHONUNBUFFERED=1
source /venv/bin/activate

# --- Custom Setup Logic ---
echo "Starting custom setup for IDM-VTON with DressCode..."

# 1. Install Dependencies
echo "Installing dependencies..."
apt-get update && apt-get install -y wget unzip git git-lfs locales sudo libgl1-mesa-glx libglib2.0-0

# 2. Smart Repository Management
DESIRED_BRANCH="dcgradio"
GITHUB_TOKEN="" # TODO make this a secret
WORKSPACE_DIR="/workspace/IDM-VTON"
REPO_URL="https://github.com/el-mashta/IDM-VTON.git"

# Configure repository URL based on available credentials
if [ -n "$GITHUB_TOKEN" ]; then
    REPO_URL="https://${GITHUB_TOKEN}@github.com/el-mashta/IDM-VTON.git"
    echo "🔐 Using authenticated GitHub access"
elif [ -n "$GITHUB_SSH_KEY" ]; then
    # Setup SSH if key is provided
    mkdir -p /root/.ssh
    echo "$GITHUB_SSH_KEY" > /root/.ssh/id_ed25519
    chmod 600 /root/.ssh/id_ed25519
    ssh-keyscan github.com >> /root/.ssh/known_hosts
    REPO_URL="git@github.com:el-mashta/IDM-VTON.git"
    echo "🔐 Using SSH key for GitHub access"
else
    # Fallback to public access (will fail for private repos)
    REPO_URL="https://github.com/el-mashta/IDM-VTON.git"
    echo "⚠️  No credentials provided - public access only"
fi


if [ -d "$WORKSPACE_DIR" ]; then
    echo "IDM-VTON directory exists, checking current state..."
    cd "$WORKSPACE_DIR"
    
    # Check if it's a git repository
    if [ -d ".git" ]; then
        CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
        CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "unknown")
        
        echo "Current branch: $CURRENT_BRANCH"
        echo "Current remote: $CURRENT_REMOTE"
        echo "Desired branch: $DESIRED_BRANCH"
        
        if [ "$CURRENT_BRANCH" = "$DESIRED_BRANCH" ] && [ "$CURRENT_REMOTE" = "$REPO_URL" ]; then
            echo "✅ Already on correct branch and remote, pulling latest changes..."
            git pull origin "$DESIRED_BRANCH"
            NEED_CLONE=false
        else
            echo "🔄 Branch or remote mismatch, need to re-clone..."
            NEED_CLONE=true
        fi
    else
        echo "🔄 Not a git repository, need to re-clone..."
        NEED_CLONE=true
    fi
else
    echo "📁 IDM-VTON directory doesn't exist, need to clone..."
    NEED_CLONE=true
fi

if [ "$NEED_CLONE" = true ]; then
    echo "Removing existing directory and cloning fresh..."
    # Change to a safe directory BEFORE removing the workspace
    cd /workspace  
    rm -rf "$WORKSPACE_DIR"
    echo "Cloning your forked IDM-VTON repository ($DESIRED_BRANCH branch)..."
    git clone -b "$DESIRED_BRANCH" "$REPO_URL" "$WORKSPACE_DIR"
    
    # Check if clone was successful
    if [ ! -d "$WORKSPACE_DIR" ]; then
        echo "❌ Git clone failed, exiting..."
        exit 1
    fi
fi

cd "$WORKSPACE_DIR"

# Verify branch and DressCode features
echo "Current branch: $(git branch --show-current)"
echo "Checking for DressCode support..."

# Check for DressCode-specific indicators
DRESSCODE_FOUND=false

# Check for the DressCode label map
if grep -q "DressCode label map" gradio_demo/app.py 2>/dev/null; then
    DRESSCODE_FOUND=true
fi

# Check for category-specific choices in dropdown
if grep -q '"upper_body", "lower_body", "dresses"' gradio_demo/app.py 2>/dev/null; then
    DRESSCODE_FOUND=true
fi

# Check for get_agnostic_mask function (DressCode specific)
if grep -q "def get_agnostic_mask" gradio_demo/app.py 2>/dev/null; then
    DRESSCODE_FOUND=true
fi

# Check for IDM-VTON-DC model loading
if grep -q "yisol/IDM-VTON-DC" gradio_demo/app.py 2>/dev/null; then
    DRESSCODE_FOUND=true
fi

if [ "$DRESSCODE_FOUND" = true ]; then
    echo "✅ DressCode support detected"
    # Show specific features found
    echo "   - Features detected:"
    grep -q "DressCode label map" gradio_demo/app.py 2>/dev/null && echo "     ✓ DressCode label mapping"
    grep -q '"upper_body", "lower_body", "dresses"' gradio_demo/app.py 2>/dev/null && echo "     ✓ Category dropdown with 3 options"
    grep -q "def get_agnostic_mask" gradio_demo/app.py 2>/dev/null && echo "     ✓ Category-specific masking function"
    grep -q "yisol/IDM-VTON-DC" gradio_demo/app.py 2>/dev/null && echo "     ✓ IDM-VTON-DC model support"
    grep -q "load_models_for_category" gradio_demo/app.py 2>/dev/null && echo "     ✓ Dynamic model loading"
else
    echo "❌ DressCode support not found in app.py"
    echo "   Debug: Checking for basic Gradio interface..."
    if grep -q "gr\." gradio_demo/app.py 2>/dev/null; then
        echo "     ✓ Gradio interface found"
    else
        echo "     ❌ No Gradio interface found"
    fi
fi

# Verify required files exist
echo "Checking required files..."
if [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt found"
else
    echo "❌ requirements.txt missing"
fi

if [ -f "download_models.py" ]; then
    echo "✅ download_models.py found"
else
    echo "❌ download_models.py missing"
fi

# 3. Install Python Dependencies
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Installing basic dependencies..."
    pip install gradio torch torchvision transformers diffusers accelerate huggingface-hub
fi

# 4. Download and Place Checkpoints from the Hugging Face Space
echo "Downloading and placing required checkpoints..."

# Check if checkpoints already exist
CHECKPOINTS_EXIST=true
if [ ! -f "ckpt/densepose/model_final_162be9.pkl" ]; then
    CHECKPOINTS_EXIST=false
fi
if [ ! -f "ckpt/humanparsing/parsing_atr.onnx" ]; then
    CHECKPOINTS_EXIST=false
fi
if [ ! -f "ckpt/openpose/ckpts/body_pose_model.pth" ]; then
    CHECKPOINTS_EXIST=false
fi

if [ "$CHECKPOINTS_EXIST" = true ]; then
    echo "✅ Checkpoints already exist, skipping download..."
else
    echo "📥 Downloading missing checkpoints..."
    # Change to /tmp for git lfs operations to avoid directory issues
    cd /tmp
    git lfs install
    git clone https://huggingface.co/spaces/yisol/IDM-VTON /tmp/hf-space

    # Ensure checkpoint directories exist
    mkdir -p "$WORKSPACE_DIR/ckpt/densepose"
    mkdir -p "$WORKSPACE_DIR/ckpt/humanparsing"
    mkdir -p "$WORKSPACE_DIR/ckpt/openpose/ckpts"

    # Move the real checkpoints into your application's ckpt directory
    cp -r /tmp/hf-space/ckpt/densepose/* "$WORKSPACE_DIR/ckpt/densepose/" 2>/dev/null || echo "DensePose checkpoints not found"
    cp -r /tmp/hf-space/ckpt/humanparsing/* "$WORKSPACE_DIR/ckpt/humanparsing/" 2>/dev/null || echo "Human parsing checkpoints not found"
    cp -r /tmp/hf-space/ckpt/openpose/ckpts/* "$WORKSPACE_DIR/ckpt/openpose/ckpts/" 2>/dev/null || echo "OpenPose checkpoints not found"

    # Clean up the temporary clone
    rm -rf /tmp/hf-space
    
    # Return to workspace directory
    cd "$WORKSPACE_DIR"
fi

# 5. Download Core Hugging Face Models using your script
echo "Downloading core diffusion models from Hugging Face..."
cd /workspace/IDM-VTON
if [ -f "download_models.py" ]; then
    python download_models.py
else
    echo "download_models.py not found, downloading manually..."
    python -c "
from huggingface_hub import snapshot_download
print('Downloading IDM-VTON models...')
snapshot_download(repo_id='yisol/IDM-VTON')
print('Downloading IDM-VTON-DC models...')
snapshot_download(repo_id='yisol/IDM-VTON-DC')
print('Models downloaded successfully')
"
fi

# 6. User and SSH Configuration
echo "Creating and configuring users sharif and stephanie..."
# -- Sharif's Setup --
if ! id "sharif" &>/dev/null; then
    useradd -m -s /bin/bash sharif && echo 'sharif:dull-remind-modern-below' | chpasswd
    mkdir -p /home/sharif/.ssh
    echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE8reWxn0Px1daHxku3K1jj1dEu4bCghmoGUx1Y3h4ye sharif@keystone" > /home/sharif/.ssh/authorized_keys
    chown -R sharif:sharif /home/sharif/.ssh
    chmod 700 /home/sharif/.ssh && chmod 600 /home/sharif/.ssh/authorized_keys
    echo "✅ User sharif created"
else
    echo "✅ User sharif already exists"
fi

# -- Stephanie's Setup --
if ! id "stephanie" &>/dev/null; then
    useradd -m -s /bin/bash stephanie && echo 'stephanie:entire-step-aim-course' | chpasswd
    mkdir -p /home/stephanie/.ssh
    echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBKTh9yyDDJmGzdIZOwnB9RZ17SBUvTSOs15pDt/kVsS stephaniekim@Stephanies-MacBook-Pro.local" > /home/stephanie/.ssh/authorized_keys
    chown -R stephanie:stephanie /home/stephanie/.ssh
    chmod 700 /home/stephanie/.ssh && chmod 600 /home/stephanie/.ssh/authorized_keys
    echo "✅ User stephanie created"
else
    echo "✅ User stephanie already exists"
fi

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