import cv2
import numpy as np
import os
import re

def save_coverage_map(coverage_count, txt_folder):
    txt_folder = txt_folder + r"\coverage_map" + ".jpg"
    cv2.imwrite(txt_folder, coverage_count, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def save_main_puzzle(puzzle, txt_folder, t1, t2):
    cv2.imwrite(txt_folder + r"\solution_" + str(t1) + "_" + str(t2) + "_.jpg", puzzle, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

def save_image_as_jpeg_pieces(images, output_path, titles):
    # Convert the image to JPEG format and save it to the output path
    for i in range(0, len(images)):
        cv2.imwrite(output_path + "\\" + "piece_" + str(titles[i]) + "_relative" + ".jpg", images[i], [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def show_covergae(covergae):
    covergae1 = cv2.normalize(covergae, None, 0, 255, cv2.NORM_MINMAX)
    covergae2 = cv2.applyColorMap(covergae1, cv2.COLORMAP_JET)
    cv2.imshow("Coverage Map", covergae2)
    return covergae2


def show_images_by_place(images_by_place, used):
    txt = " - Piece"
    for i in range(0, len(images_by_place)):
        txt1 = str(used[i]) + txt
        cv2.imshow(txt1, images_by_place[i])


def extract_puzzle_affine_number(path):
    # Define the regex pattern to match the desired part
    pattern = r"puzzle_(affine|homography)_(\d+)"

    # Search for the pattern in the path string
    match = re.search(pattern, path)

    if match:
        # Extract the numeric part from the match object
        number = int(match.group(2))
        return number
    else:
        return None


def determine_puzzle_type(path):
    if "puzzle_affine" in path:
        return 0
    elif "puzzle_homography" in path:
        return 1
    else:
        raise ValueError("Invalid puzzle path: {}".format(path))


def resize_with_padding(image, new_height, new_width):
    # Get the height and width of the input image
    height, width = image.shape[:2]

    # Calculate the difference in height and width between the original image and desired size
    height_diff = new_height - height
    width_diff = new_width - width

    # Calculate the padding amounts for top, bottom, left, and right
    top = height_diff // 2
    bottom = height_diff - top
    left = width_diff // 2
    right = width_diff - left

    # Create a black image with the desired height and width
    padded_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

    # Insert the original image into the center of the padded image
    padded_image[top:top + height, left:left + width, :] = image

    return padded_image


def read_matrix_from_file(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        # Read the lines from the file
        lines = file.readlines()

        # Initialize an empty matrix
        matrix = np.zeros((3, 3), dtype=np.float32)

        # Loop through the lines and extract the matrix elements
        for i in range(3):
            elements = lines[i].strip().split('\t')
            for j in range(3):
                matrix[i, j] = float(elements[j])

        return matrix


def txt_W_H_loader(path):
    all_files = os.listdir(path)
    txt_files = [file for file in all_files if file.endswith(".txt")]
    filename = txt_files[0]
    # Define a regular expression pattern to match the numbers
    pattern = r".*_(\d+)__H_(\d+)__W_(\d+)_.*\.txt"
    match = re.match(pattern, filename)
    # Extract the numbers from the match object
    H = match.group(2)
    W = match.group(3)

    return txt_files[0], H, W


def load_images_from_folder(folder, Width, Hieght):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            # img = resize_with_padding(img, Width, Hieght)
            images.append(img)
    return images


def convert_to_grayscale(images):
    grayscale_images = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale_images.append(gray)
    return grayscale_images


def match_and_estimate_affine(matcher, ransac_threshold, num_iterations, kp1, kp2, des1, des2, ratio_threshold=0.75,
                              MIN_MATCH_COUNT=10):
    # Match keypoints using the matcher
    matches = matcher.knnMatch(des1, des2, 2)

    # Filter out unreliable matches using ratio-test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    if len(good_matches) > MIN_MATCH_COUNT:
        # Sort good matches by distance
        good_matches = sorted(good_matches, key=lambda x: x.distance)

        # Extract matched keypoints and convert them to float32 arrays
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        affine = None
        # Estimate affine transformation using RANSAC
        if (len(dst_pts) >= 3 and len(src_pts) >= 3):
            affine, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC,
                                             ransacReprojThreshold=ransac_threshold, maxIters=num_iterations)
    else:
        return None

    return affine


def match_and_estimate_homography(matcher, ransac_threshold, num_iterations, kp1, kp2, des1, des2, ratio_threshold=0.75,
                                  MIN_MATCH_COUNT=10):
    # Match keypoints using the matcher
    matches = matcher.knnMatch(des1, des2, k=2)

    # Filter out unreliable matches using ratio-test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    if len(good_matches) > MIN_MATCH_COUNT:
        # Sort good matches by distance
        good_matches = sorted(good_matches, key=lambda x: x.distance)

        # Extract matched keypoints and convert them to float32 arrays
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate homography using RANSAC
        homography = None
        if (len(dst_pts) >= 4 and len(src_pts) >= 4):
            homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold, maxIters=num_iterations)
    else:
        # print("No good matches found")
        return None
    return homography


def solve_puzzle_affine(images_folder, num_iterations, ransac_threshold, ratio_threshold, MIN_MATCH_COUNT, Hieght,
                        Width, delta):
    # Load images from folder
    images = load_images_from_folder(images_folder, Width, Hieght)
    grayscale_images = convert_to_grayscale(images)
    images[0] = cv2.warpPerspective(images[0], matrix1,
                                    (Hieght, Width), borderMode=cv2.BORDER_TRANSPARENT)

    # Initialize SIFT detector, descriptor, and matcher
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher()

    panorama = images[0]
    # Apply affine matrices to merge the remaining images
    ratio_arr = []
    used = [1]

    for i in range(0, len(images)):
        ratio_arr.append(ratio_threshold)

    # Create a coverage_count
    coverage_count = np.zeros((Width, Hieght), dtype=np.uint8)
    non_black_pixels = np.any(panorama != [0, 0, 0], axis=-1)
    coverage_count[non_black_pixels] += 1
    images_by_place = [images[0].copy()]

    c1 = len(used)
    for j in range(0, len(images)):
        for i in range(1, len(images)):
            # print(len(used))
            if images[i] is not None:
                gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
                kp1, des1 = sift.detectAndCompute(grayscale_images[i], None)
                kp2, des2 = sift.detectAndCompute(gray_panorama, None)
                affine = match_and_estimate_affine(matcher, ransac_threshold, num_iterations, kp1, kp2, des1, des2,
                                                   ratio_threshold=ratio_arr[i], MIN_MATCH_COUNT=MIN_MATCH_COUNT)
                if affine is not None:
                    im1 = cv2.warpAffine(images[i], affine, (Hieght, Width), borderMode=cv2.BORDER_TRANSPARENT)
                    non_black_pixels = np.any(im1 != [0, 0, 0], axis=-1)
                    coverage_count[non_black_pixels] += 1
                    images_by_place.append(im1.copy())
                    mask = panorama <= 1
                    # Use the mask to update the pixels in panorama array
                    panorama[mask.all(axis=2)] = im1[mask.all(axis=2)]
                    images[i] = None
                    grayscale_images[i] = None
                    used.append(i + 1)

        print("Loop %d, used num = %d" % (j, len(used)))
        if (len(used) - c1 == 0):
            break
        else:
            c1 = len(used)
    images[0] = None
    non_used = []
    for i in range(len(images)):
        if images[i] is not None:
            non_used.append(i + 1)
    if len(non_used) != 0:
        print("Puzzle is not full!, percent of used images = ", 100 * (len(images) - len(non_used)) / len(images), "%")
        print("Used images:", used)
        print("Total number of used images:", len(used))
        print("Missing images:", non_used)
    else:
        print("The puzzle is full, the order of the used images: ", used)

    return panorama, images_by_place, coverage_count, used, len(non_used)


def solve_puzzle_homography(images_folder, num_iterations, ransac_threshold, ratio_threshold, Hieght, Width, delta,
                            MIN_MATCH_COUNT):
    # Load images from folder
    images = load_images_from_folder(images_folder, Width, Hieght)
    grayscale_images = convert_to_grayscale(images)
    images[0] = cv2.warpPerspective(images[0], matrix1,
                                    (Hieght, Width), borderMode=cv2.BORDER_TRANSPARENT)
    # Initialize SIFT detector, descriptor, and matcher
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher()

    panorama = images[0]
    # Apply homography matrices to merge the remaining images
    ratio_arr = []
    used = [1]

    for i in range(0, len(images)):
        ratio_arr.append(ratio_threshold)

    # Create a coverage_count
    coverage_count = np.zeros((Width, Hieght), dtype=np.uint8)
    non_black_pixels = np.any(panorama != [0, 0, 0], axis=-1)
    coverage_count[non_black_pixels] += 1
    images_by_place = [images[0].copy()]

    c1 = len(used)
    for j in range(0, len(images)):
        for i in range(1, len(images)):
            # print(len(used))
            if images[i] is not None:
                gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
                kp1, des1 = sift.detectAndCompute(grayscale_images[i], None)
                kp2, des2 = sift.detectAndCompute(gray_panorama, None)
                homography = match_and_estimate_homography(matcher, ransac_threshold, num_iterations, kp1, kp2, des1,
                                                           des2, ratio_threshold=ratio_arr[i],
                                                           MIN_MATCH_COUNT=MIN_MATCH_COUNT)
                if homography is not None:
                    im1 = cv2.warpPerspective(images[i], homography, (Hieght, Width), borderMode=cv2.BORDER_TRANSPARENT)
                    non_black_pixels = np.any(im1 != [0, 0, 0], axis=-1)
                    coverage_count[non_black_pixels] += 1
                    images_by_place.append(im1.copy())
                    mask = panorama <= 1
                    # Use the mask to update the pixels in panorama array
                    panorama[mask.all(axis=2)] = im1[mask.all(axis=2)]
                    images[i] = None
                    grayscale_images[i] = None
                    used.append(i + 1)
        print("Loop %d, used num = %d" % (j, len(used)))
        if (len(used) - c1 == 0):
            break
        else:
            c1 = len(used)
    images[0] = None
    non_used = []
    for i in range(len(images)):
        if images[i] is not None:
            non_used.append(i + 1)
    if len(non_used) != 0:
        print("Puzzle is not full!, percent of used images = ", 100 * (len(images) - len(non_used)) / len(images), "%")
        print("Used images:", used)
        print("Total number of used images:", len(used))
        print("Missing images:", non_used)
    else:
        print("The puzzle is full, the order of the used images: ", used)
    return panorama, images_by_place, coverage_count, used, len(non_used)


if __name__ == '__main__':
    num_iterations = 1000000000
    delta = 1

    ransac_arr_Homography = [5.0, 5.0, 5.0, 5.0, 8.0, 5.0, 5.0, 5.0, 5.0, 5.0, 9.0]
    ratio_arr_Homography = [0.7, 0.7, 0.7, 0.65, 0.6, 0.75, 0.6, 0.7, 0.7, 0.82]
    min_match_Homography = [10, 10, 10, 10, 10, 10, 10, 10, 15, 22]

    ransac_arr_Affine = [5.0, 5.0, 5.0, 5.0, 5.0, 2.0, 8.0, 5.0, 9.0, 9.0]
    ratio_arr_Affine = [0.7, 0.7, 0.7, 0.7, 0.7, 0.95, 0.75, 0.75, 0.775, 0.84]
    min_match_Affine = [10, 10, 10, 10, 10, 10, 20, 15, 12, 17]

    numm = 10
    images_folder = r"C:\Users\USER\Desktop\CompVis\HW1\PuzzleRASNAC\puzzles\puzzles\puzzle_homography_" + str(numm) + "\pieces"  # Provide the path to the folder containing puzzle piece images
    txt_folder = r"C:\Users\USER\Desktop\CompVis\HW1\PuzzleRASNAC\puzzles\puzzles\puzzle_homography_" + str(numm)
    txt_name, H1, W1 = txt_W_H_loader(txt_folder)
    matrix1 = read_matrix_from_file(txt_folder + "\\" + txt_name)
    if (determine_puzzle_type(images_folder) == 0):
        puzzle_number = extract_puzzle_affine_number(images_folder)
        title = "Puzzle Affine - " + str(puzzle_number)
        print("Affine puzzle")
        ransac_threshold = ransac_arr_Affine[puzzle_number-1]
        ratio_threshold = ratio_arr_Affine[puzzle_number-1]
        MIN_MATCH_COUNT = min_match_Affine[puzzle_number-1]
        result, images_by_place, coverage_count, used, non_used_num = solve_puzzle_affine(images_folder, num_iterations=num_iterations,
                                                                      ransac_threshold=ransac_threshold,
                                                                      ratio_threshold=ratio_threshold,
                                                                      Hieght=int(W1), Width=int(H1), delta=delta,
                                                                      MIN_MATCH_COUNT=MIN_MATCH_COUNT)  # Change the desired number of iterations here
        coverage_count1 = show_covergae(coverage_count)
        # show_images_by_place(images_by_place, used)
        # output_path = r"C:\Users\USER\Desktop\CompVis\HW1\PuzzleRASNAC\puzzles\results" + r"\puzzle_affine_" + str(puzzle_number)
        # save_main_puzzle(result, output_path, len(used), len(used) + non_used_num)
        # save_image_as_jpeg_pieces(images_by_place, output_path, used)
        # save_coverage_map(coverage_count1, output_path)
    else:
        puzzle_number = extract_puzzle_affine_number(images_folder)
        title = "Puzzle Homography - " + str(puzzle_number)
        print("Homography puzzle")
        ransac_threshold = ransac_arr_Homography[puzzle_number - 1]
        ratio_threshold = ratio_arr_Homography[puzzle_number - 1]
        MIN_MATCH_COUNT = min_match_Homography[puzzle_number - 1]
        result, images_by_place, coverage_count, used, non_used_num = solve_puzzle_homography(images_folder, num_iterations=num_iterations,
                                         ransac_threshold=ransac_threshold, ratio_threshold=ratio_threshold,
                                         Hieght=int(W1), Width=int(H1), delta=delta,
                                         MIN_MATCH_COUNT=MIN_MATCH_COUNT)  # Change the desired number of iterations here
        coverage_count1 = show_covergae(coverage_count)
        # show_images_by_place(images_by_place, used)
        # output_path = r"C:\Users\USER\Desktop\CompVis\HW1\PuzzleRASNAC\puzzles\results" + r"\puzzle_homography_" + str(puzzle_number)
        # save_main_puzzle(result, output_path, len(used), len(used) + non_used_num)
        # save_image_as_jpeg_pieces(images_by_place, output_path, used)
        # save_coverage_map(coverage_count1, output_path)
    cv2.imshow(title, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
