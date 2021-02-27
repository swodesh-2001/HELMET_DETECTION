import cv2

cap = cv2.VideoCapture("Resources/vid2.mp4")
fig = cv2.createBackgroundSubtractorMOG2()
bike_cascade = cv2.CascadeClassifier("Resources/two_wheeler.xml")

while True:
    sucess , img = cap.read()
    fig.apply(img)
    if (type(img) == type(None)):
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bikes = bike_cascade.detectMultiScale(gray, 1.4 ,1)
    for (x, y, w, h) in bikes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 215), 2)
    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
