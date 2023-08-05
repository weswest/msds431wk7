package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"reflect"

	"github.com/e-XpertSolutions/go-iforest/iforest"
	randomforest "github.com/malaschitz/randomForest"
	"github.com/petar/GoMNIST"
)

// This is related to GoMNIST
func printImage(image GoMNIST.RawImage) {
	scaleFactor := 255.0 / 8.0
	nRow := 28
	nCol := 28

	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			// Get the pixel value at the current position
			pixel := image[i*nCol+j]

			// Scale the pixel value
			scaledPixel := int(math.Round(float64(pixel) / scaleFactor))

			// Make sure that only 0 scales to 0
			if pixel != 0 && scaledPixel == 0 {
				scaledPixel = 1
			}

			// Print a space if the pixel value is 0, otherwise print the scaled pixel value
			if scaledPixel == 0 {
				fmt.Print(" ")
			} else {
				fmt.Print(scaledPixel)
			}
		}
		// Start a new line after each row
		fmt.Println()
	}
}

func convertMNISTForModeling(images []GoMNIST.RawImage) [][]float64 {
	var floatImages [][]float64

	for _, image := range images {
		var floatImage []float64
		for _, pixel := range image {
			floatImage = append(floatImage, float64(pixel))
		}
		floatImages = append(floatImages, floatImage)
	}

	return floatImages
}

func main() {
	rand.Seed(431)

	//////////////////////////
	// GoMNIST time			//
	//////////////////////////
	fmt.Println("GoMNIST time")
	fmt.Println("GoMNIST time")
	fmt.Println("GoMNIST time")

	// Load the dataset
	train, test, err := GoMNIST.Load("./data")
	if err != nil {
		fmt.Println(err)
	}

	// This code returns the train and test MNIST.Set types
	// Set has NRow, NCol, Images ([]RawImage), Labels ([]Label)

	fmt.Println("Rows: ", train.NRow, test.NRow)
	fmt.Println("Columns: ", train.NCol, test.NCol)
	// Print the first image in the train and test set
	//	fmt.Println(train.Labels[0], train.Images[0])
	//	fmt.Println(test.Labels[0], test.Images[0])
	//	fmt.Println(reflect.TypeOf(train.Images[0]))

	inputData := convertMNISTForModeling(train.Images)

	//////////////////////////
	// iForest time			//
	//////////////////////////

	// input parameters
	treesNumber := 100000
	subsampleSize := 256
	outliersRatio := 0.01
	//routinesNumber := 10

	//model initialization
	forest := iforest.NewForest(treesNumber, subsampleSize, outliersRatio)

	//training stage - creating trees
	forest.Train(inputData)

	//testing stage - finding anomalies
	//Test or TestParaller can be used, concurrent version needs one additional
	// parameter
	forest.Test(inputData)
	//	forest.TestParallel(inputData, routinesNumber)

	//after testing it is possible to access anomaly scores, anomaly bound
	// and labels for the input dataset

	//	threshold := forest.AnomalyBound
	iForestAnomalyScores := forest.AnomalyScores
	//	labelsTest := forest.Labels

	//	fmt.Println("Anomaly scores: ", iForestAnomalyScores)
	fmt.Println("Anomaly scores length: ", len(iForestAnomalyScores))
	fmt.Println("Anomaly scores type: ", reflect.TypeOf(iForestAnomalyScores))
	fmt.Println("Anomaly score for first item: ", iForestAnomalyScores[0])

	//////////////////////////
	// RandomForest time	//
	//////////////////////////

	fmt.Println("Starting RandomForest")
	rForest := randomforest.IsolationForest{X: inputData}
	rForest.Train(100000)
	fmt.Println("rForest len", len(rForest.Results))
	fmt.Println("rForest.Results type: ", reflect.TypeOf(rForest.Results))
	fmt.Println("rForest.Results[0] type: ", reflect.TypeOf(rForest.Results[0]))
	fmt.Println("rForest.Results[0]", rForest.Results[0])

	//////////////////////////
	// CSV time				//
	//////////////////////////

	// Create the CSV file
	file, err := os.Create("results/goIForestScores.csv")
	if err != nil {
		fmt.Println("Could not create file:", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write the header
	if err := writer.Write([]string{"idx", "label", "iForestAnomalyScore"}); err != nil {
		fmt.Println("Error writing header:", err)
		return
	}

	// Write the rows
	for idx, score := range iForestAnomalyScores {
		label := train.Labels[idx]
		row := []string{fmt.Sprint(idx + 1), fmt.Sprint(label), fmt.Sprintf("%.6f", score)} // Adjust the format as needed
		if err := writer.Write(row); err != nil {
			fmt.Println("Error writing row:", err)
			return
		}
	}
}
