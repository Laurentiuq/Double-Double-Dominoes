import cv2 as cv
import numpy as np
import os
from additional.universal import show_image, show_image_no_resize
import pickle

def extrage_info(imagine):
    """
    Aplica operatori pentru a elimina pixelii albi ce nu fac parte din chenar
    """
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv.erode(imagine, kernel, iterations=2)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(erosion, cv.MORPH_OPEN, kernel, iterations=2)
    return opening


def crop_essential_image(imagine):
    """Creeaza imaginea care contine doar chenarul pe care se plaseaza piesele"""
    img = imagine.copy()
    img = img[1160:2912, 732:2456].copy()
    return img


def create_mask_and_res(img):
    """Creeaza o masca binara si o imagine rezultata din aplicarea mastii pe imaginea initiala"""

    # Cropeaza imaginea pentru a obtine chenarul din mijloc
    img_croped = crop_essential_image(img)

    # Parametrii pentru crearea mastii
    low_yellow = (49, 0, 0)
    high_yellow = (120, 255, 255)

    img_hsv = cv.cvtColor(img_croped, cv.COLOR_BGR2HSV)

    # Masca binara din hsv
    mask_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
    # Imaginea rezultata
    res = cv.bitwise_and(img_croped, img_croped, mask=mask_hsv)
    return res, mask_hsv


def extrage_colturi_imagine_binara(mask_hsv, res):
    """Extrage colturile imaginii binare pentru a folosi apoi perspectivTransform.
    Reutrneaza si imaginaea reazultata prin evidentierea colturilor"""
    mask_hsv_copy = mask_hsv.copy()
    mask_hsv_copy = extrage_info(mask_hsv_copy)
    # show_image('mask_hsv_copy', mask_hsv_copy)
    top_left = 0
    bottom_right = 0
    top_right = 0
    bottom_left = 0

    # ----------------------------------------------------------------------------
    # COD LABORATOR
    canny_mask = cv.Canny(mask_hsv_copy, 400, 400)
    # show_image('edges', canny_mask)
    contours, _ = cv.findContours(canny_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0

    for i in range(len(contours)):
        if (len(contours[i]) > 3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    # cv.circle(res,tuple(top_left),20,(0,0,255),-1)
    # cv.circle(res,tuple(top_right),20,(0,0,255),-1)
    # cv.circle(res,tuple(bottom_left),20,(0,0,255),-1)
    # cv.circle(res,tuple(bottom_right),20,(0,0,255),-1)

    return res, top_left, top_right, bottom_right, bottom_left


def transform_to_square_perspective(mask_yellow_hsv, res):
    """Transforma imaginea data ca parametru intr-o imagine patrata cu dimensiune fixa"""
    width_F = 1635
    height_F = 1635
    res_crop, top_left, top_right, bottom_right, bottom_left = extrage_colturi_imagine_binara(mask_yellow_hsv, res)
    bias = 0
    board = np.array([top_left - bias, top_right + bias, bottom_right + bias, bottom_left - bias], dtype=np.float32)
    destination_board = np.array([[0, 0], [width_F, 0], [width_F, height_F], [0, height_F]], dtype=np.float32)
    M = cv.getPerspectiveTransform(board, destination_board)
    warped = cv.warpPerspective(res, M, (width_F, height_F))
    return warped


# scopul e ca pentru o zona de la marginea tablei sa nu se adauge prea mult negru
def apply_margins_white_mask(image):
    """Adauga margini albe la imagine(in caz ca mai raman zone negre)"""
    image_copy = image.copy()
    make_white_margins_mask = np.zeros((1635, 1635), np.uint8)
    make_white_margins_mask[0:10, :] = 255
    make_white_margins_mask[1625:1635, :] = 255
    make_white_margins_mask[:, 0:10] = 255
    make_white_margins_mask[:, 1625:1635] = 255
    image_copy = cv.bitwise_or(image_copy, make_white_margins_mask)
    return image_copy


def aplica_template(domi, templates_dir='templates'):
    """Template matching pe un domino, returneaza numele fisierului cu cel mai bun template"""
    maxi = -np.inf
    poz = -1
    for temp in os.listdir(templates_dir):
        img_template = cv.imread(os.path.join(templates_dir, temp))
        img_template = cv.cvtColor(img_template, cv.COLOR_BGR2GRAY)
        corr = cv.matchTemplate(domi, img_template, cv.TM_CCOEFF_NORMED)
        corr = np.max(corr)
        if corr > maxi:
            maxi = corr
            poz = temp
    return poz


width_F = 1635
height_F = 1635


def extrage_domino(img_path):
    # pozitiile de pe tabla pe care deja am gasit un domino
    seen = set()
    img = cv.imread(img_path)

    # Creeaza masca binara si imaginea rezultata din aplicarea mastii
    res, mask_yellow_hsv = create_mask_and_res(img)
    # Transforma imaginea cu tabla intr-o imagine de dimensiune fixa
    warpedf = transform_to_square_perspective(mask_yellow_hsv, res)
    show_image("warped", warpedf)

    # Deseneaza liniile de pe tabla
    # DACA LE SCOT NU MAI MERGE
    lines = []
    for i in range(0, width_F, 109):
        lines.append(i)
    for line in lines:
        cv.line(warpedf, (line, 0), (line, height_F), (255, 255, 255), 2)
        cv.line(warpedf, (0, line), (width_F, line), (255, 255, 255), 2)

    # Imaginea warped gri
    warped_gray = cv.cvtColor(warpedf, cv.COLOR_BGR2GRAY)
    show_image("warped_gray", warped_gray)

    # Imaginea warped binara
    warped_bin = cv.threshold(warped_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    show_image('warped_bin', warped_bin)
    warped_bin = apply_margins_white_mask(warped_bin)
    show_image('warped_bin', warped_bin)
    l_ind = -1
    linii = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    coloane = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    dic = {}
    sol = []
    for i in range(5, height_F-5, 109):
        l_ind += 1
        c_ind = -1

        for j in range(5, width_F-5, 109):
            c_ind += 1
            # print(linii[l_ind], coloane[c_ind])
            domino = warped_bin[i:i + 109, j:j + 109]
            # show_image_no_resize('domino', domino)
            sum_domino = np.sum(domino)
            # print(l_ind, c_ind)
            # print(f"{linii[c_ind] + 1}, {coloane[l_ind]}: {sum_domino}")
            dic[(linii[l_ind], coloane[c_ind])] = sum_domino
            if sum_domino > 1700000:
                if (linii[l_ind], coloane[c_ind]) == (7, 'I'):
                    print("ESTE 7I")
                    colored_patch = warpedf[i:i + 109, j:j + 109]
                    possible_covered = np.sum(colored_patch[:, :, :], 1)
                    print(f"Possible covered: {np.sum(possible_covered)}")
                    print(np.sum(possible_covered))
                    if np.sum(possible_covered) < 6000000 and 170 <= dic[(7, 'I')] // 10000 <= 185:
                        continue

                if (linii[l_ind], coloane[c_ind]) in seen:
                    continue

                sol.append((linii[l_ind], coloane[c_ind]))
                seen.add((linii[l_ind], coloane[c_ind]))
                cv.rectangle(warpedf, (j, i), (j + 109, i + 109), (0, 255, 0), 5)

    # Valoarea de pe celula 7I
    print("7IL:", dic[(7, 'I')] // 10000)
    show_image('warped', warpedf)
    return sol


def find_nr_circles(template_dir='templates'):
    """Gaseste numarul de cercuri de pe un dominourile folosite ca templates"""
    dict_nr_circles = {}
    # for temp in ["template53.jpg"]:
    for temp in os.listdir(template_dir):
        img_template = cv.imread(os.path.join(template_dir, temp))
        img_template = cv.cvtColor(img_template, cv.COLOR_BGR2GRAY)
        img_template = cv.medianBlur(img_template, 5)
        # show_image_no_resize('img_template', img_template)
        circles = cv.HoughCircles(img_template, cv.HOUGH_GRADIENT, 1, 20, param1=300, param2=28, minRadius=8,
                                  maxRadius=100)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(img_template, center, 1, (0, 100, 100), 3)
                # circle outline
                cv.circle(img_template, center, 5, (255, 0, 255), 3)
            dict_nr_circles[temp] = len(circles[0, :])
        else:
            dict_nr_circles[temp] = 0
        print(temp)
        # show_image_no_resize('img_template', img_template)
    return dict_nr_circles




def testeaza_solutie(rez_dir='rezultate', sol_dir='antrenare'):
    """Testeaza solutia si afiseaza numarul de greseli pentru fiecare cerinta."""
    nr_gresite_cerinta1 = 0
    nr_gresite_cerinta2 = 0
    nr_gresite_cerinta3 = 0
    for file in os.listdir(rez_dir):
        # if file[:3] == "sol":
        fisier_sol = open(os.path.join(rez_dir, file), 'r')
        fisier_corect = open(sol_dir + "/" + file, 'r')
        sol = fisier_sol.readlines()
        corect = fisier_corect.readlines()
        fisier_sol.close()
        fisier_corect.close()
        if sol[0].split()[0] == corect[0].split()[0] and sol[1].split()[0] == corect[1].split()[0]:
            # print("Corect: ", file)
            if sol[0].split()[1] == corect[0].split()[1] and sol[1].split()[1] == corect[1].split()[1]:
                pass
                # print("Corect C2: ", file)
            else:
                print("Gresit!! C2: ", file)
                print(sol[0].split()[1] == corect[0].split()[1] and sol[1].split()[1] == corect[1].split()[1])
                nr_gresite_cerinta2 += 1
        else:
            print("Gresit C1: ", file)
            nr_gresite_cerinta1 += 1
            nr_gresite_cerinta2 += 1
        if sol[2].strip() != corect[2].strip():
            # print("Gresit C3: ", file)
            nr_gresite_cerinta3 += 1
    print(f"Numar gresite 1: {nr_gresite_cerinta1}")
    print(f"Numar gresite 2: {nr_gresite_cerinta2}")
    print(f"Numar gresite 3: {nr_gresite_cerinta3}")

def verifica_sg_domino(de_verf):
    """Verifica un singur domino"""
    see = set()
    linii = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    coloane = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    for el in linii:
        for el2 in coloane:
            see.add((el + 1, el2))
    for poz in de_verf:
        see.remove(poz)
    return see

def creeaza_dictionar_nr_circles():
    """Gaseste si salveaza numarul de pe o jumatate de domino"""
    dict_nr_circles = find_nr_circles('templates')
    dict_nr_circles['template53'] = 4
    with open('dict_nr_circles.pkl', 'wb') as file:
        pickle.dump(dict_nr_circles, file)

def genereaza_templates(img_path, nr_template=1):
    template_dir = 'templates'
    width_F = 1635
    height_F = 1635
    seen = set()
    img = cv.imread(img_path)
    # show_image('img', img)
    res, mask_yellow_hsv = create_mask_and_res(img)
    warpedf = transform_to_square_perspective(mask_yellow_hsv, res)
    show_image("warped", warpedf)
    lines = []

    for i in range(0, width_F, 109):
        lines.append(i)

    for line in lines:
        cv.line(warpedf, (line, 0), (line, height_F), (255, 255, 255), 2)
        cv.line(warpedf, (0, line), (width_F, line), (255, 255, 255), 2)
    last_7i = 0
    # show_image('warped', warpedf)
    warped_gray = cv.cvtColor(warpedf, cv.COLOR_BGR2GRAY)
    show_image("warped_gray", warped_gray)

    # TODO:
    # Create a 1635x1635 black binary image with white margins

    warped_bin = cv.threshold(warped_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    warped_bin = apply_margins_white_mask(warped_bin)
    show_image('warped_bin', warped_bin)
    l_ind = -1
    c_ind = -1
    linii = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    coloane = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    dic = {}
    sol = []
    for i in range(0, width_F, 109):
        l_ind += 1
        c_ind = -1

        for j in range(0, height_F, 109):
            c_ind += 1
            domino = warped_bin[j:j + 109, i:i + 109]
            domino_gray = warped_gray[j:j + 109, i:i + 109]
            # show_image("domino", domino)
            sum_domino = np.sum(domino)
            # print(l_ind, c_ind)
            # print(f"{linii[c_ind] + 1}, {coloane[l_ind]}: {sum_domino}")
            dic[(linii[c_ind] + 1, coloane[l_ind])] = sum_domino

            if sum_domino > 1700000:
                if (linii[c_ind] + 1, coloane[l_ind]) == (7, 'I'):
                    print("ESTE 7I")
                    colored_patch = warpedf[j:j + 109, i:i + 109]
                    possible_covered = np.sum(colored_patch[:,:,:], 1)
                    print(f"Possible covered: {np.sum(possible_covered)}")
                    if np.sum(possible_covered) < 600000:
                        continue

                if (linii[c_ind] + 1, coloane[l_ind]) in seen:
                    continue
                cv.imwrite(os.path.join(template_dir, f'template{nr_template}' + '.jpg'), domino_gray)
                nr_template += 1
                sol.append((linii[c_ind] + 1, coloane[l_ind]))
                seen.add((linii[c_ind] + 1, coloane[l_ind]))
                cv.rectangle(warpedf, (i, j), (i + 109, j + 109), (0, 255, 0), 5)

    print(sol)
    print(dic[(7, 'I')]//10000)
    show_image('warped', warpedf)
    return sol
# 6214587
