package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	framework "tensorflow/tensorflow/core/framework"
	serving "tensorflow/tensorflow_serving"

	google_protobuf "github.com/golang/protobuf/ptypes/wrappers"
	"google.golang.org/grpc"
)

func main() {
	servingAddress := flag.String("serving-address", "localhost:8500", "tensorflow serving address")
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Printf("Usage: %s --serving-address=<SERVING_ADDRESS> IMAGE_FILE\n", os.Args[0])
		os.Exit(1)
	}

	imagePath, err := filepath.Abs(flag.Arg(0))
	if err != nil {
		log.Fatalf("image file not exist: %s\n", flag.Arg(0))
	}

	imageFile, err := os.Open(imagePath)
	if err != nil {
		log.Fatalf("open file failure: %v\n", err)
	}
	defer imageFile.Close()
	img, name, err := image.Decode(imageFile)
	if err != nil {
		log.Fatalf("decode image failure: %v\n", err)
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
			imageFloats[i*3+0] = float32(r>>8)
			imageFloats[i*3+1] = float32(g>>8)
			imageFloats[i*3+2] = float32(b>>8)
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

	conn, err := grpc.Dial(*servingAddress, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("dial serving address failure: %v", err)
	}
	defer conn.Close()

	client := serving.NewPredictionServiceClient(conn)

	resp, err := client.Predict(context.Background(), request)
	if err != nil {
		log.Fatalf("request serving failure: %v", err)
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
		log.Fatalf("image create failure: %v\n", err)
	}
	err = png.Encode(newImageFile, resImage)
	if err != nil {
		log.Fatalf("image encode failure: %v\n", err)
	}

	content, err := json.Marshal(resp)
	ioutil.WriteFile("segmap.json", content, 0644)
}
