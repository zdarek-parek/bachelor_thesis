# Application

## How to download
The application is an executable file for Windows, and to run it, the user has to download the file to their PC or laptop. The application creates a folder with results, which means that the application has to be stored in a directory with permission to write. 
Download the application from the link: https://gitlab.mff.cuni.cz/dariako/bakalarka/-/blob/master/ClassArt_app/ClassArt.zip?ref_type=heads


## How to run
To run the application, the user must click the icon of the application. It might take a while before the application starts. Once the main window is rendered, the user can click any buttons.


## How to use
1. **CSV file input**: CSV file with at least two fields:
"item" - ID of the reproduction in the database, "imageAddr" - address of an image, might be an URL address (for example https://digi.ub.uni-heidelberg.de/iiif/3/dkd1898_1899%3A000_p.jpg/128,169,1683,544/max/0/default.jpg) of an image in a digital library, or an address in a file system of a locally stored image. The application processes only fields "item" and "imageAddr" and ignores the rest of the fields if present.

2. **Directory input**: Directory that contains directories that contain images. The following schema illustrates expected input structure:

input_dir/

    ├── dir1/
    │   ├── img1.jpeg
    │   ├── img2.jpeg
    ├── dir2/
        ├── img3.jpeg
        └── img4.jpeg

The input in the above example is input_dir.
The application works with PNG, JPEG and JPG images.
The applicateion ignores files in input_dir and works only with folders.
The application ignores non-images in dir1, dir2 etc. and works only with images of the formats stated above.

3. **Output**: CSV file with fields: "item", "imageAddr", "class1", "prob1" - the name and corresponding probability of the most probable class, "class2", "prob2" - the name and corresponding probability of the second most probable class, "class3", "prob3" - the name and corresponding probability of the third most probable class.
In case of CSV input (1.) the fields "item" and "imageAddr" are copied from the input CSV file, in case of directory input (2.) "item" is a name of the image, "imageAddr" is a path to the image in the file system.
After all the input is processed the application shows a Finish window.
The result CSV files will be stored in the folder with the name "result" that is created in the same folder where the application is. Each output CSV file is named in the following way: the name of the input CSV file or folder appended with "_result" suffix. Input CSV file - "query1.csv", corresponding output file - "query1_result.csv". Input folder - "image_folder1", corresponding output CSV file - "image_folder1_result.csv".

4. **Errors**:

**Directory input**: In case of errors while working with images, the application stops processing images in the current inside directory and closes the result CSV file once it encounters an error. After that, it proceeds to process next inside directory, if there is any.

**CSV file input**:In case of errors in the input CSV file, the application processes the file until it encounters an error. After that, it stops processing the input file, closes the result CSV file with the content the application was able to produce before it encountered the error, and proceeds to process the next input CSV file, if there is any.


# Code

To run the code in the ClassArt_app directory


1. Clone or copy the directory.
2. Create python environment in the directory where the ClassArt_app is with command python -m venv C:\path\to\new\virtual\environment
3. Install 
    * PySide6 (pip install PySide6) 
    * numpy (pip install numpy)
    * cv2 (pip install opencv-python) 
    * torchvision, torch (pip3 install torch torchvision torchaudio) - for Windows with CPU, the code was tested only on laptops without GPU, the application is built with CPU version
    * PIL (pip install pillow)
    * skimage (pip install scikit-image)
4. Download weights from the link https://gitlab.mff.cuni.cz/dariako/bakalarka/-/blob/master/ClassArt_app/torch_model.pt?ref_type=heads and put the file torch_model.pt in the directory with the code files (the same way it is in https://gitlab.mff.cuni.cz/dariako/bakalarka/-/tree/master/ClassArt_app?ref_type=heads).
5. Download bakalarkaAppAvatar.ico from the link https://gitlab.mff.cuni.cz/dariako/bakalarka/-/blob/master/ClassArt_app/bakalarkaAppAvatar.ico?ref_type=heads and put the file bakalarkaAppAvatar.ico in the directory with the code files (the same way it is in https://gitlab.mff.cuni.cz/dariako/bakalarka/-/tree/master/ClassArt_app?ref_type=heads).
6. Run the ui_qt.py file.

# Link to the project
https://gitlab.mff.cuni.cz/dariako/bakalarka