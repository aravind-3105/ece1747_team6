# Main.cpp

Parameters:
1. input_folder
2. output_background
3. frame_file
4. output_foreground
5. output_preprocessed
6. output_labeled

Steps:
1. Read the input folder and get the list of files.
2. Read the background image.
3. Read the frame file.
4. Run GMG algorithm to get the foreground mask.
5. Extract the foreground from the frame.
6. Preprocess the foreground image.
7. Connecte components labeling on the preprocessed image.
8. Generate binary boundary image.
9. Compute fourier descriptors.
10. Compute HOG descriptors.
11. Save the output images and descriptors.
12. Compute SVM predicitons usuing the descriptors separately and get the final prediction.
13. Save the final prediction.
14. Repeat the process for all the frames in the input folder.