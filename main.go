package main

import (
	"fmt"
	"image"
	"image/color"

	"gocv.io/x/gocv"
)

var model = "MobileNetSSD_deploy.caffemodel"
var config = "MobileNetSSD_deploy.prototxt"

var CLASSES = []string{"background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"}

func main() {
	modelNet := gocv.ReadNet(model, config)
	modelNet.SetPreferableBackend(gocv.NetBackendVKCOM)
	modelNet.SetPreferableTarget(gocv.NetTargetVulkan)

	deviceID := "0"

	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()

	webcam.Set(gocv.VideoCaptureFrameWidth, 720)
	webcam.Set(gocv.VideoCaptureFrameHeight, 480)

	window := gocv.NewWindow("Capture Window")
	defer window.Close()

	img := gocv.NewMat()
	defer img.Close()

	resized := gocv.NewMat()
	defer resized.Close()

	fmt.Printf("Start reading device: %v\n", deviceID)
	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Device closed: %v\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}

		img.ConvertTo(&resized, gocv.MatTypeCV32F)

		blob := gocv.BlobFromImage(resized, 0.007843, image.Pt(300, 300), gocv.NewScalar(127.5, 127.5, 127.5, 0), false, false)

		modelNet.SetInput(blob, "data")

		blob.Close()

		detection := modelNet.Forward("detection_out")

		for i := 0; i < detection.Total(); i += 7 {
			confidence := detection.GetFloatAt(0, i+2)

			if confidence > 0.4 {
				id := int(detection.GetFloatAt(0, i+1))
				left := int(detection.GetFloatAt(0, i+3) * float32(img.Cols()))
				top := int(detection.GetFloatAt(0, i+4) * float32(img.Rows()))
				right := int(detection.GetFloatAt(0, i+5) * float32(img.Cols()))
				bottom := int(detection.GetFloatAt(0, i+6) * float32(img.Rows()))
				r := image.Rect(left, top, right, bottom)
				gocv.Rectangle(&img, r, color.RGBA{0, 0, 255, 0}, 2)
				gocv.PutText(&img, fmt.Sprintf("%s:%f", CLASSES[id], confidence), image.Pt(r.Min.X, r.Min.Y), gocv.FontItalic, 1.0, color.RGBA{0, 0, 255, 0}, 2)
			}
		}

		window.IMShow(img)

		if window.WaitKey(1) == 27 {
			break
		}
	}
}
