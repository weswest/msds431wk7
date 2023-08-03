package main

import (
	"fmt"
	"math"
	"reflect"
	"sort"

	"github.com/e-XpertSolutions/go-iforest/iforest"
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
	fmt.Println(train.Labels[0], train.Images[0])
	fmt.Println(test.Labels[0], test.Images[0])
	fmt.Println(reflect.TypeOf(train.Images[0]))

	printImage(train.Images[0])
	printImage(test.Images[0])

	//////////////////////////
	// iForest time			//
	//////////////////////////

	inputData := convertMNISTForModeling(train.Images)

	// input parameters
	treesNumber := 100
	subsampleSize := 256
	outliersRatio := 0.01
	routinesNumber := 10

	//model initialization
	forest := iforest.NewForest(treesNumber, subsampleSize, outliersRatio)

	//training stage - creating trees
	forest.Train(inputData)

	//testing stage - finding anomalies
	//Test or TestParaller can be used, concurrent version needs one additional
	// parameter
	forest.Test(inputData)
	forest.TestParallel(inputData, routinesNumber)

	//after testing it is possible to access anomaly scores, anomaly bound
	// and labels for the input dataset

	//	threshold := forest.AnomalyBound
	anomalyScores := forest.AnomalyScores
	//	labelsTest := forest.Labels

	fmt.Println("Anomaly scores: ", anomalyScores)
	fmt.Println("Anomaly scores length: ", len(anomalyScores))
	fmt.Println("Anomaly scores type: ", reflect.TypeOf(anomalyScores))
	fmt.Println("Anomaly score for first item: ", anomalyScores[0])

	const numBands = 10
	const numLabels = 10
	var table [numLabels][numBands]int
	var totalPerLabel [numLabels]int

	// Loop over all images in the dataset
	for idx, score := range anomalyScores {
		label := train.Labels[idx]
		band := int(score * numBands)

		// If score is exactly 1.0, it should fall into the last band (index 9), not a new band
		if band == numBands {
			band = numBands - 1
		}

		table[label][band]++
		totalPerLabel[label]++
	}

	// Print the table
	for label, counts := range table {
		fmt.Printf("Label %d: ", label)
		for _, count := range counts {
			percentage := float64(count) / float64(totalPerLabel[label]) * 100
			fmt.Printf("%.2f%% ", percentage)
		}
		fmt.Println()
	}
	//////////////////////////
	// Anomalies vs Normals	//
	//////////////////////////

	type ScoredImage struct {
		Score float64
		Image GoMNIST.RawImage
	}

	// Group all the images by label
	imagesByLabel := make(map[GoMNIST.Label][]ScoredImage)
	for idx, score := range anomalyScores {
		label := train.Labels[idx]
		image := train.Images[idx]

		imagesByLabel[label] = append(imagesByLabel[label], ScoredImage{
			Score: score,
			Image: image,
		})
	}

	// For each label, find the two images with the lowest anomaly scores and
	// one image with the highest anomaly score, then print them
	for label, images := range imagesByLabel {
		// Sort images by score
		sort.Slice(images, func(i, j int) bool {
			return images[i].Score < images[j].Score
		})

		fmt.Printf("Label %d:\n", label)

		if len(images) >= 2 {
			fmt.Println("Two images with the lowest anomaly scores:")
			printImage(images[0].Image)
			printImage(images[1].Image)
		}

		if len(images) >= 1 {
			fmt.Println("One image with the highest anomaly score:")
			printImage(images[len(images)-1].Image)
		}
	}

	//to get information about new instances pass them to the Predict function
	// to speed up computation use concurrent version of Predict
	// var newData [][]float64
	// newData = loadData("someNewInstances")
	// labels, scores := forest.Predict(newData)
}
