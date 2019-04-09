package main

import (
	"flag"
	"fmt"
	_ "image/jpeg"
	"log"
	"os"
	"path/filepath"
)

func main() {
	servingAddress := flag.String("serving-address", "localhost:8500", "tensorflow serving address")
	modelName := flag.String("model", "deeplab_v3", "model name: deeplab_v3, ssd_mobilenet_v2_coco")
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Printf("Usage: %s --serving-address=<SERVING_ADDRESS> IMAGE_FILE\n", os.Args[0])
		os.Exit(1)
	}

	imagePath, err := filepath.Abs(flag.Arg(0))
	if err != nil {
		log.Fatalf("image file not exist: %s\n", flag.Arg(0))
	}

	switch *modelName {
	case "deeplab_v3":
		err = CallDeeplabService(*servingAddress, imagePath)
	case "ssd_mobilenet_v2_coco":
		err = CallSsdMoblieNetCoco(*servingAddress, imagePath)
	default:
		log.Fatalf("unknown model: %s\n", *modelName)
	}

	if err != nil {
		log.Fatalf("call service failure: %v\n", err)
	}

}
