import cv2

SELECTION_COLOUR = (222,0,222)
WINDOW_NAME = "Select regions with a mouse"
IMAGE_FILE = 'img.png'
OUTPUT_FILE = "img_roi.png"

def mouseClickEvent(event, x, y, flags, data):
    image, points = data
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(x) + ',' +
                str(y), (x,y), font,
                0.5, (255, 0, 0), 1)
        print(f'Point 1: {x}, {y}')
        
    elif event == cv2.EVENT_LBUTTONUP:
        points.append((x, y))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(x) + ',' +
                str(y), (x,y), font,
                0.5, (255, 0, 0), 1)
        print(f'Point 2: {x}, {y}')
        cv2.rectangle(image, points[-2], points[-1], SELECTION_COLOUR, 2)
        cv2.imshow(WINDOW_NAME, image)

def show_mouse_select(image_filename):
    orig = cv2.imread(image_filename)
    image = orig.copy()
    cv2.namedWindow(WINDOW_NAME)

    points = []
    cv2.setMouseCallback(WINDOW_NAME, mouseClickEvent, (image, points))

    while True:
        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKey(1)
        if key == ord('q'): break

    # Output points and save image
    if len(points)>1:
        # print ("Points:")
        for i in range(0,len(points),2):
            a, b = points[i], points[i+1]
            # print (f'A: {min(a,b)}, B:{max(a,b)}')
        
        cv2.imwrite(OUTPUT_FILE, image)
        print ("Saved to:", OUTPUT_FILE)

    cv2.destroyAllWindows()

if __name__=="__main__":
    
    image = cv2.imread(IMAGE_FILE)
    h,w,c = image.shape
    print(f'Image Height: {h}, Image Width: {w}')

    show_mouse_select(IMAGE_FILE)