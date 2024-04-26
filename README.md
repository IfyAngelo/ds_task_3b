# Project Name

## Description

This project implements an image classification and object detection system using the You Only Look Once (YOLO) algorithm. The system is capable of detecting objects in images and assigning labels based on the presence of memory-related content.

## Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Files](#files)
4. [Contributing](#contributing)
5. [License](#license)

## Installation <a name="installation"></a>

To run this project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   ```

2. Navigate to the project directory:

   ```bash
   cd your-repo
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage <a name="usage"></a>

To use the YOLO image classification and object detection system, follow these steps:

1. Ensure you have annotated data available in the `annotated_data` directory. This directory should contain two JSON files (`labels_memory.json` and `labels_no-memory.json`) with annotations for memory-related content.

2. Prepare your image data by organizing memory and no-memory images into separate directories (`memory` and `no_memory`).

3. Run the `yolo.py` script to start the Flask server:

   ```bash
   python yolo.py
   ```

4. Once the server is running, you can send POST requests to the `/detect_memory` endpoint to detect memory-related content in images.

5. Use the `test.py` script to test the system with an image file:

   ```bash
   python test.py
   ```

   Replace `"image_path"` in `test.py` with the path to your image file.

## Files <a name="files"></a>

- `yolo.py`: Python script containing the main implementation of the YOLO image classification and object detection system.
- `test.py`: Python script for testing the system by sending a POST request with an image file.
- `annotated_data`: Directory containing JSON files with annotated data for memory-related content.
- `memory`: Directory containing memory-related image data.
- `no_memory`: Directory containing no-memory image data.
- `requirements.txt`: Text file listing all the required Python packages and their versions.

## Contributing <a name="contributing"></a>

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License <a name="license"></a>

This project is licensed under the [MIT License](LICENSE).
