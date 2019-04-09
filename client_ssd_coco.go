package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io/ioutil"
	"log"
	"math/rand"
	"os"

	framework "tensorflow/tensorflow/core/framework"
	serving "tensorflow/tensorflow_serving"

	"github.com/fogleman/gg"
	google_protobuf "github.com/golang/protobuf/ptypes/wrappers"
	"google.golang.org/grpc"
)

func CallSsdMoblieNetCoco(servingAddress string, imagePath string) error {
	imageFile, err := os.Open(imagePath)
	if err != nil {
		return fmt.Errorf("open file failure: %v", err)
	}
	defer imageFile.Close()
	img, name, err := image.Decode(imageFile)
	if err != nil {
		return fmt.Errorf("decode image failure: %v", err)
	}

	log.Printf("image format = %v\n", name)

	height := img.Bounds().Dy()
	width := img.Bounds().Dx()
	log.Printf("image bounds: %+v, h = %d, w = %d\n", img.Bounds(), height, width)

	imageData := make([]int32, height*width*3)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			c := img.At(x, y)
			r, g, b, _ := c.RGBA()
			i := y*width + x
			imageData[i*3+0] = int32(r >> 8)
			imageData[i*3+1] = int32(g >> 8)
			imageData[i*3+2] = int32(b >> 8)
		}
	}

	request := &serving.PredictRequest{
		ModelSpec: &serving.ModelSpec{
			Name:          "ssd_mobilenet_v2_coco",
			SignatureName: "serving_default",
			VersionChoice: &serving.ModelSpec_Version{
				Version: &google_protobuf.Int64Value{
					Value: int64(1),
				},
			},
		},
		Inputs: map[string]*framework.TensorProto{
			"inputs": &framework.TensorProto{
				Dtype: framework.DataType_DT_UINT8,
				TensorShape: &framework.TensorShapeProto{
					Dim: []*framework.TensorShapeProto_Dim{
						&framework.TensorShapeProto_Dim{
							Size: int64(1),
						},
						&framework.TensorShapeProto_Dim{
							Size: int64(height),
						},
						&framework.TensorShapeProto_Dim{
							Size: int64(width),
						},
						&framework.TensorShapeProto_Dim{
							Size: int64(3),
						},
					},
				},
				IntVal: imageData,
			},
		},
	}

	conn, err := grpc.Dial(servingAddress, grpc.WithInsecure())
	if err != nil {
		return fmt.Errorf("dial serving address failure: %v", err)
	}
	defer conn.Close()

	client := serving.NewPredictionServiceClient(conn)

	resp, err := client.Predict(context.Background(), request)
	if err != nil {
		return fmt.Errorf("request serving failure: %v", err)
	}

	detectionClasses := resp.Outputs["detection_classes"] // DT_FLOAT (-1, 100)
	numDetections := resp.Outputs["num_detections"]       // DT_FLOAT (-1)
	detectionBoxes := resp.Outputs["detection_boxes"]     // DT_FLOAT (-1, 100, 4)
	detectionScores := resp.Outputs["detection_scores"]   // DT_FLOAT (-1, 100)

	response, err := json.MarshalIndent(resp.Outputs, "", "  ")
	ioutil.WriteFile("ssd_response.json", response, 0644)

	num := numDetections.FloatVal[0]
	log.Printf("num_detections = %.2f\n", num)

	colorList := []string{
		"0B6C7E",
		"1F89A7",
		"F2D7AE",
		"E8952E",
		"7CA321",
		"DBCF64",
		"E9B343",
		"DF474E",
		"5D2A36",
		"3D363A",
	}

	cachedColors := make(map[int]string)
	getColor := func(id int) string {
		if c, ok := cachedColors[id]; ok {
			return c
		}
		c := colorList[rand.Intn(len(colorList))]
		cachedColors[id] = c
		return c
	}

	// draw bounding boxes
	dc := gg.NewContext(width, height)
	dc.DrawImage(img, 0, 0)
	dc.SetLineWidth(3.0)

	loadLabels := func() []string {
		fin, err := os.Open("coco_labels.txt")
		if err != nil {
			log.Printf("read labels failure: %v\n", err)
			return nil
		}
		defer fin.Close()

		labels := []string{"UNKNOWN"}
		scanner := bufio.NewScanner(fin)
		for scanner.Scan() {
			labels = append(labels, scanner.Text())
		}
		return labels
	}

	labels := loadLabels()
	totalLabels := int(detectionClasses.TensorShape.Dim[1].Size)
	log.Printf("total labels: %d\n", totalLabels)
	for i := 0; i < totalLabels; i++ {
		score := detectionScores.FloatVal[i]
		if score > 0 {
			class := int(detectionClasses.FloatVal[i])
			y0 := float64(detectionBoxes.FloatVal[i*4+0] * float32(height))
			x0 := float64(detectionBoxes.FloatVal[i*4+1] * float32(width))
			y1 := float64(detectionBoxes.FloatVal[i*4+2] * float32(height))
			x1 := float64(detectionBoxes.FloatVal[i*4+3] * float32(width))
			log.Printf("i = %d, score = %.3f, class = %d, rect = (%.2f, %.2f)-(%.2f, %.2f)\n",
				i, score, class, x0, y0, x1, y1)
			//
			classname := ""
			if class < len(labels) {
				classname = labels[class]
			}
			dc.Push()
			dc.SetHexColor(getColor(class))
			title := fmt.Sprintf("class: %d [%s], score: %.6f", class, classname, score)
			dc.DrawString(title, x0, y0-5-float64(rand.Int31n(2)*10))
			dc.DrawRectangle(x0, y0, x1-x0, y1-y0)
			dc.Stroke()
			dc.Pop()
		}
	}

	dc.SavePNG("ssd_result.png")

	return nil
}

// API Description
//
// {
//   "model_spec": {
//     "name": "ssd_mobilenet_v2_coco",
//     "signature_name": "",
//     "version": "1"
//   },
//   "metadata": {
//     "signature_def": {
//       "signature_def": {
//         "serving_default": {
//           "inputs": {
//             "inputs": {
//               "dtype": "DT_UINT8",
//               "tensor_shape": {
//                 "dim": [
//                   {
//                     "size": "-1",
//                     "name": ""
//                   },
//                   {
//                     "size": "-1",
//                     "name": ""
//                   },
//                   {
//                     "size": "-1",
//                     "name": ""
//                   },
//                   {
//                     "size": "3",
//                     "name": ""
//                   }
//                 ],
//                 "unknown_rank": false
//               },
//               "name": "image_tensor:0"
//             }
//           },
//           "outputs": {
//             "detection_classes": {
//               "dtype": "DT_FLOAT",
//               "tensor_shape": {
//                 "dim": [
//                   {
//                     "size": "-1",
//                     "name": ""
//                   },
//                   {
//                     "size": "100",
//                     "name": ""
//                   }
//                 ],
//                 "unknown_rank": false
//               },
//               "name": "detection_classes:0"
//             },
//             "num_detections": {
//               "dtype": "DT_FLOAT",
//               "tensor_shape": {
//                 "dim": [
//                   {
//                     "size": "-1",
//                     "name": ""
//                   }
//                 ],
//                 "unknown_rank": false
//               },
//               "name": "num_detections:0"
//             },
//             "detection_boxes": {
//               "dtype": "DT_FLOAT",
//               "tensor_shape": {
//                 "dim": [
//                   {
//                     "size": "-1",
//                     "name": ""
//                   },
//                   {
//                     "size": "100",
//                     "name": ""
//                   },
//                   {
//                     "size": "4",
//                     "name": ""
//                   }
//                 ],
//                 "unknown_rank": false
//               },
//               "name": "detection_boxes:0"
//             },
//             "detection_scores": {
//               "dtype": "DT_FLOAT",
//               "tensor_shape": {
//                 "dim": [
//                   {
//                     "size": "-1",
//                     "name": ""
//                   },
//                   {
//                     "size": "100",
//                     "name": ""
//                   }
//                 ],
//                 "unknown_rank": false
//               },
//               "name": "detection_scores:0"
//             }
//           },
//           "method_name": "tensorflow/serving/predict"
//         }
//       }
//     }
//   }
// }
