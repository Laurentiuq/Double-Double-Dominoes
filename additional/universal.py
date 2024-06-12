import cv2 as cv

def mouse_callback(event, x, y, flags, param):
    """Shows mouse coordinates at the cursor position when clicked"""
    if event == cv.EVENT_LBUTTONDOWN:
        print(f"Mouse coordinates: ({x}, {y})")

def show_img_win_resize(img):
    cv.namedWindow("img", cv.WINDOW_KEEPRATIO)
    cv.resizeWindow("img", 900, 800)
    cv.imshow("img", img)
    cv.setMouseCallback("img", mouse_callback)
    cv.waitKey(0)
    cv.destroyAllWindows()

def show_image(title, img):
    """Shows image resized to fit the screen, mouse callback on"""
    image = img.copy()
    image = cv.resize(image, (0, 0), fx=0.4, fy=0.35)
    # print(f"Show image size: {image.shape}")
    cv.imshow(title, image)
    cv.setMouseCallback(title, mouse_callback)
    cv.waitKey(0)
    cv.destroyAllWindows()


def show_image_no_resize(title, image):
    """Shows without resizing, mouse callback on"""
    cv.imshow(title, image)
    cv.setMouseCallback(title, mouse_callback)
    cv.waitKey(0)
    cv.destroyAllWindows()
