# VanGuardian
# VanGuardian: School Vehicle Security and Management

VanGuardian is an innovative software solution designed to elevate the security and streamline the management of school vehicles, offering a robust system that integrates computer vision and neural networks. The project focuses on real-time detection, identification, and verification of school vans, license plates, and specific decals known as prefixes.

## Features:

### 1. Real-time Detection Using YOLO
   - Leveraging YOLO (You Only Look Once) architecture, VanGuardian ensures efficient and accurate detection of school vans from live video feeds.

### 2. License Plate and Decal Recognition
   - Utilizes neural networks to identify license plates and prefixes on detected vans, enhancing the system's ability to extract critical information.

### 3. OCR (Optical Character Recognition)
   - Employs OCR technology to extract alphanumeric characters from license plates and prefixes, facilitating seamless verification.

### 4. List Verification
   - Cross-references the extracted license plates and prefixes with a predefined list, providing instant feedback on the approval status of the school vehicle.

## Getting Started:

### Prerequisites:
   - Ensure the installation of the required dependencies: `cv2`, `ultralytics`, `PIL`, `pytesseract`, `pandas`, and `numpy`.

### Usage:
   1. Load pre-trained YOLO models for vans, license plates, and prefixes.
   2. Execute the main script by running the following command in your terminal or command prompt:

```bash
python app.py
```
## Project Background:

This project serves as the capstone for the completion of the Bachelor's degree in Computer Science by Luiz Passos. It showcases the practical application of computer vision and neural networks in addressing real-world challenges, specifically in the domain of school vehicle security and management.

## Performance Metrics:

VanGuardian strives for optimal performance, with a focus on real-time processing. The average processing time for vehicle detection and verification is consistently monitored and displayed during runtime.

## Contributing:

We welcome contributions and feedback! Feel free to fork the repository, open issues, and submit pull requests to enhance the functionality of VanGuardian.

## License:

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

## Acknowledgments:

We appreciate the open-source community and the contributions of various libraries and frameworks that make VanGuardian possible.
