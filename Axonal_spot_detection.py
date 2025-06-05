# Import Libraries
import cv2 as cv
import imutils
import numpy as np
import matplotlib.pyplot as plt
import csv
# import pandas as pd

for i in ['file_name_1','file_name_2']:
# for i in ['C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/gfp/gfp_live_z2_ortho.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/gfp/gfp_live_z3_ortho.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/gfp/gfp_live_z4_ortho.czi-C=0.avi']:
# for i in ['C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/1mM_NMN/gfp_optimem_1mM_NMN_1.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/1mM_NMN/gfp_optimem_1mM_NMN_2.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/1mM_NMN/gfp_optimem_1mM_NMN_3.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/1mM_NMN/gfp_optimem_1mM_NMN_4.czi-C=0.avi']:
# for i in ['C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/100_nM_NMN/gfp_optimem_100uM_NMN_1.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/100_nM_NMN/gfp_optimem_100uM_NMN_2.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/100_nM_NMN/gfp_optimem_100uM_NMN_3.czi-C=0.avi']:
# for i in ['C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/anisomycin/gfp_optimem_40uM_anisomycin_z1_ortho.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/anisomycin/gfp_optimem_40uM_anisomycin_z2_ortho.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/anisomycin/gfp_optimem_40uM_anisomycin_z3_ortho.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/anisomycin/gfp_optimem_40uM_anisomycin_z4_ortho.czi-C=0.avi']:
# for i in ['C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/control/control_no_gfp_live_z1_ortho.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/control/control_no_gfp_live_z2_ortho.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/control/control_no_gfp_live_z3_ortho.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/control/control_no_gfp_live_z4_ortho.czi-C=0.avi']:
# for i in ['C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/ngf/gfp_live_ngf_100ngml_z1_ortho.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/ngf/gfp_live_ngf_100ngml_z2_ortho.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/ngf/gfp_live_ngf_100ngml_z3_ortho.czi-C=0.avi', 'C:/Users/user/PycharmProjects/pythonProject/cropped_avi_files/ngf/gfp_live_ngf_100ngml_z4_ortho.czi-C=0.avi']:
    cap = cv.VideoCapture(i)  # Capturing video from video file
    print('New video')
    x_axis = []
    avg_intensity_per_frame = []

    while (cap.isOpened()):
        ret, frame = cap.read()
        # cv.imshow('frame', frame)
        # length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        # print(length)
        if ret == True:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert frames from BGR to Grayscale
            ret, thresh = cv.threshold(gray, 30, 255, cv.THRESH_BINARY)  # Setting threshold value to 4 and converting all higher values to 255 i.e. white
            # print('thresh', thresh)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # Finding Contours of hotspots
            for i, cnt in enumerate(contours):
                cv.drawContours(frame, contours, i, (0, 0, 255), 1)  # Drawing Contours framewise

            lst_intensities = []
            avg_per_contour = []
            lst_intensities_bgnd = []
            avg_per_contour_bgnd = []
            avg_per_contour_diff = []

            pts = np.where(thresh == 255)  # Grabbing x and y coords of white pixels
            # print('pts', pts)
            # cv.imshow('thresh', thresh)
            pts_bgnd = np.where(thresh == 0)

            for k in range(len(pts_bgnd[0])):
                lst_intensities_bgnd.append(gray[pts_bgnd[0][k], pts_bgnd[1][k]])  # Taking the coordinates of background and putting them in the grayscale image to get intensity
            # print('lst_intensities_bgnd', lst_intensities_bgnd)
            # length = len(lst_intensities_bgnd)
            # print('length', length)
            avg_per_contour_bgnd = np.mean(lst_intensities_bgnd)
            # print('avg_per_contour_bgnd', avg_per_contour_bgnd)
            # print('avg_intensity_per_frame', avg_intensity_per_frame)

            for j in range(len(pts[0])):
                lst_intensities.append(gray[pts[0][j], pts[1][j]])  # Taking the coordinates of hotspots from above and putting them in the grayscale image
            avg_per_contour_diff = lst_intensities - avg_per_contour_bgnd
            # length1 = len(lst_intensities)
            avg_per_contour = sum(avg_per_contour_diff)
            # print('length1', length1)
            # print('lst_intensities', lst_intensities)
            # print('avg_per_contour_diff', avg_per_contour_diff)
            # print('avg_per_contour', avg_per_contour)
            avg_intensity_per_frame.append(avg_per_contour)
            time = cap.get(cv.CAP_PROP_POS_FRAMES) / cap.get(cv.CAP_PROP_FPS)
            x_axis.append(time)
            frame = imutils.resize(frame, 330, 280, inter=cv.INTER_CUBIC)
            cv.imshow('Detected', frame)
            # plt.plot(x_axis, avg_intensity_per_frame)
            if cv.waitKey(50) & 0xFF == ord('q'):
                break
        else:
            break

    # plt.xlabel('time (in sec)')
    # plt.ylabel('avg_intensity_per_frame')

    print('avg_intensity_per_frame', avg_intensity_per_frame)
    print("x-axis", x_axis)

    label = np.array(['gfp(old)', '1mM_NMN', '100_nM_NMN', 'anisomycin', 'control', 'ngf(old)', 'gfp-ngf(new)'])
    # label = np.array(['1', '2', '3', '4', '5', '6'])
    plt.plot(x_axis, avg_intensity_per_frame)
    plt.legend(label)
    headerList = ['Time', 'Average Intensity']
    new = ['New Video']
    row = [x_axis, avg_intensity_per_frame]
    with open('output.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(new)
        writer.writerow(headerList)
        writer.writerows([row])

plt.show()
cap.release()
cv.destroyAllWindows()
