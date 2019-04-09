package main

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"io/ioutil"
	"log"
	"os"

	framework "tensorflow/tensorflow/core/framework"
	serving "tensorflow/tensorflow_serving"

	google_protobuf "github.com/golang/protobuf/ptypes/wrappers"
	"google.golang.org/grpc"
)

func CallDeeplabService(servingAddress string, imagePath string) error {
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

	imageFloats := make([]float32, height*width*3)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			c := img.At(x, y)
			r, g, b, _ := c.RGBA()
			i := y*width + x
			imageFloats[i*3+0] = float32(r >> 8)
			imageFloats[i*3+1] = float32(g >> 8)
			imageFloats[i*3+2] = float32(b >> 8)
		}
	}

	request := &serving.PredictRequest{
		ModelSpec: &serving.ModelSpec{
			Name:          "deeplab_v3",
			SignatureName: "predict_images",
			VersionChoice: &serving.ModelSpec_Version{
				Version: &google_protobuf.Int64Value{
					Value: int64(1),
				},
			},
		},
		Inputs: map[string]*framework.TensorProto{
			"height": &framework.TensorProto{
				Dtype:  framework.DataType_DT_INT32,
				IntVal: []int32{int32(height)},
			},
			"width": &framework.TensorProto{
				Dtype:  framework.DataType_DT_INT32,
				IntVal: []int32{int32(width)},
			},
			"images": &framework.TensorProto{
				Dtype: framework.DataType_DT_FLOAT,
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
				FloatVal: imageFloats,
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

	segmap := resp.Outputs["segmentation_map"]
	shape := segmap.GetTensorShape()
	imageData := segmap.Int64Val
	newHeight := int(shape.Dim[1].Size)
	newWidth := int(shape.Dim[2].Size)
	resImage := image.NewGray(image.Rect(0, 0, newWidth, newHeight))
	log.Printf("result shape = %+v, h = %d, w = %d\n", shape, newHeight, newWidth)

	for y := 0; y < newHeight; y++ {
		for x := 0; x < newWidth; x++ {
			resImage.SetGray(x, y, color.Gray{Y: uint8(imageData[y*newWidth+x] * 20)})
		}
	}

	newImageFile, err := os.Create("segmap.png")
	if err != nil {
		return fmt.Errorf("image create failure: %v", err)
	}
	err = png.Encode(newImageFile, resImage)
	if err != nil {
		return fmt.Errorf("image encode failure: %v", err)
	}

	content, err := json.Marshal(resp)
	ioutil.WriteFile("segmap.json", content, 0644)

	return nil
}

// API Description:
// 
// {
//   "model_spec": {
//     "name": "deeplab_v3",
//     "signature_name": "",
//     "version": "1"
//   },
//   "metadata": {
//     "signature_def": {
//       "signature_def": {
//         "predict_images": {
//           "inputs": {
//             "height": {
//               "dtype": "DT_INT32",
//               "tensor_shape": {
//                 "dim": [],
//                 "unknown_rank": true
//               },
//               "name": "Placeholder:0"
//             },
//             "images": {
//               "dtype": "DT_FLOAT",
//               "tensor_shape": {
//                 "dim": [
//                   {
//                     "size": "1",
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
//               "name": "x:0"
//             },
//             "width": {
//               "dtype": "DT_INT32",
//               "tensor_shape": {
//                 "dim": [],
//                 "unknown_rank": true
//               },
//               "name": "Placeholder_1:0"
//             }
//           },
//           "outputs": {
//             "segmentation_map": {
//               "dtype": "DT_INT64",
//               "tensor_shape": {
//                 "dim": [
//                   {
//                     "size": "1",
//                     "name": ""
//                   },
//                   {
//                     "size": "-1",
//                     "name": ""
//                   },
//                   {
//                     "size": "-1",
//                     "name": ""
//                   }
//                 ],
//                 "unknown_rank": false
//               },
//               "name": "ArgMax:0"
//             }
//           },
//           "method_name": "tensorflow/serving/predict"
//         }
//       }
//     }
//   }
// }
