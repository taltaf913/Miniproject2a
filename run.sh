#!/bin/bash

# Step 1: Clone the git repository and navigate into it
git clone https://github.com/vaibhavmaurya/aimlops_miniproject_module3.git
cd aimlops_miniproject_module3 || { echo "Directory aimlops_miniproject_module3 does not exist"; exit 1; }

# Step 2: Run the Python command to train the model
python3 sentiment_analysis_model/train_pipeline.py --config sentiment_analysis_model_api/config/config.yml || { echo "Failed to train model"; exit 1; }

# Step 3: Create a wheel file
python3 setup.py sdist bdist_wheel
wheel_file_path=$(find dist -name '*.whl') || { echo "Failed to find the .whl file"; exit 1; }
echo "Wheel file path: $wheel_file_path"

# Step 4: Navigate into bikeshare_model_api folder, create and activate Python virtual environment
cd sentiment_analysis_model_api || { echo "Directory bikeshare_model_api does not exist"; exit 1; }
python3 -m venv venv
source venv/bin/activate

# Step 5: Install dependencies from requirements.txt
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt does not exist in the current directory"
    exit 1
fi

# Step 6: Install the wheel file
pip install ../"$wheel_file_path" || { echo "Failed to install the .whl file"; exit 1; }

# Step 7: Start the Fast API server
uvicorn api:app --reload
