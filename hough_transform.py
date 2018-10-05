import numpy as np
import cv2 as cv


def polar_htp_line(input_image):
    rows = input_image.shape[0]
    columns = input_image.shape[1]

    rmax = int(np.sqrt(np.square(rows) + np.square(columns)))
    acc = np.zeros((rmax, 180))

    for x in range(columns):
        for y in range(rows):
            if input_image[y, x] == 255:
                for m in range(1, 181):
                    r = int((x+1)*np.cos((m*np.pi)/180) + (y+1) * np.sin((m*np.pi)/180))
                    if rmax > r > 0:
                        acc[r, m] = acc[r, m] + 1
    return acc, rmax


def ht_line(input_image):
    rows = input_image.shape[0]
    columns = input_image.shape[1]

    acc1 = np.zeros((rows, 91))
    acc2 = np.zeros((columns, 91))

    for x in range(columns):
        for y in range(rows):
            if input_image[y, x] == 255:
                for m in range(-45, 46):
                    b = int(round((y + 1) - np.tan((m * np.pi) / 180)) * (x+1))
                    if rows > b > 0:
                        acc1[b, m + 45 + 1] = acc1[b, m + 45 + 1] + 1
                for m in range(45, 136):
                    b = int(round((x+1)-(y+1)/np.tan((m * np.pi) / 180)))
                    if columns > b > 0:
                        acc1[b, m - 45 + 1] = acc1[b, m - 45 + 1] + 1
    return acc1, acc2


def ht_circle(input_image, r):
    rows = input_image.shape[0]
    columns = input_image.shape[1]

    acc = np.zeros((rows, columns))

    for x in range(columns):
        for y in range(rows):
            if input_image[y, x] == 255:
                for ang in range(0, 361):
                    t = (ang*np.pi)/180
                    x0 = int(round((x+1)-r*np.cos(t)))
                    y0 = int(round((y+1)-r*np.sin(t)))
                    if columns > x0 > 0 and rows > y0 > 0:
                        acc[y0, x0] = acc[y0, x0] + 1
    return acc


def draw_line(input_image, r_m_tuple):
    new_image = np.copy(input_image)
    m = -(1/(np.tan((r_m_tuple[1]*np.pi)/180)))
    c = r_m_tuple[0] * (1/(np.sin((r_m_tuple[1]*np.pi)/180)))

    xs = [q for q in range(0, new_image.shape[1])]
    ys = [int(m*q + c) for q in xs]
    for i in range(len(xs)):
        if input_image.shape[0] >= ys[i] >= 0:
            new_image[ys[i], xs[i]] = 255

    return new_image


def draw_circle(input_image, x_y_0_tuple, rad):
    new_img = np.copy(input_image)
    new_image = cv.circle(new_img, x_y_0_tuple, rad, color=125)
    return new_image

if __name__ == "__main__":
    # POLAR LINES
    # img = cv.imread("/home/vignesh/PycharmProjects/HoughTransform/images/sudoku_grid.jpg",
    #                 cv.IMREAD_GRAYSCALE)
    # img_edges = cv.Canny(img, 100, 200)
    # cv.imshow("", img_edges)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # accumulator, rm = polar_htp_line(img_edges)
    # new_img = np.copy(img)
    # for _ in range(15):
    #     max_indices = np.unravel_index(np.argmax(accumulator), (rm, 180))
    #     new_img = draw_line(new_img, max_indices)
    #     accumulator[max_indices[0], max_indices[1]] = -np.Inf
    # cv.imshow("", new_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Circles
    img = cv.imread("/home/vignesh/PycharmProjects/HoughTransform/images/CD.jpg",
                    cv.IMREAD_GRAYSCALE)
    img_edges = cv.Canny(img, 100, 200)
    cv.imshow("", img_edges)
    cv.waitKey(0)
    cv.destroyAllWindows()
    accumulator = None
    new_img = np.copy(img)
    for r in [50]:
        accumulator = ht_circle(img_edges, r)
        for _ in range(1):
            max_indices = np.unravel_index(np.argmax(accumulator), (img.shape[0], img.shape[1]))
            new_img = draw_circle(new_img, max_indices, r)
            accumulator[max_indices[0], max_indices[1]] = -np.Inf
    cv.imshow("", new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
