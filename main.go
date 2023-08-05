package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"reflect"
	"sort"

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

// This is related to normalizing the randomForest Isolation Forest implementation
func normalizeScores(scores []float64, min, max float64) []float64 {
	normalized := make([]float64, len(scores))
	for i, score := range scores {
		normalized[i] = (score - min) / (max - min)
	}
	return normalized
}

// This is related to pushing results out to the CSV file
func WriteCSV(iForestAnomalyScores map[int]float64, normalizedScores map[int]float64, labels []int) {
	file, err := os.Create("results/goIForestScores.csv")
	if err != nil {
		fmt.Println("Could not create file:", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write the header
	if err := writer.Write([]string{"idx", "label", "iForestAnomalyScore", "rForestNormalizedScore"}); err != nil {
		fmt.Println("Error writing header:", err)
		return
	}

	// Extract and sort the keys from the map
	var keys []int
	for k := range iForestAnomalyScores {
		keys = append(keys, k)
	}
	sort.Ints(keys)

	// Write the rows in index order
	for _, idx := range keys {
		label := labels[idx]
		iForestScore := iForestAnomalyScores[idx]
		normalizedScore := normalizedScores[idx]
		row := []string{
			fmt.Sprint(idx + 1),
			fmt.Sprint(label),
			fmt.Sprintf("%.6f", iForestScore),
			fmt.Sprintf("%.6f", normalizedScore),
		}
		if err := writer.Write(row); err != nil {
			fmt.Println("Error writing row:", err)
			return
		}
	}
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
	fmt.Println("First Train label: ", train.Labels[0])
	printImage(train.Images[0])

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
	treesNumber := 100
	subsampleSize := 256
	outliersRatio := 0.01
	//	routinesNumber := 100

	//model initialization
	forest := iforest.NewForest(treesNumber, subsampleSize, outliersRatio)

	//training stage - creating trees
	fmt.Println("Starting training")
	forest.Train(inputData)

	//testing stage - finding anomalies
	//Test or TestParaller can be used, concurrent version needs one additional
	// parameter
	fmt.Println("Starting testing")
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

	// This code prints a rough histogram for each label
	const numBands = 10
	const numLabels = 10
	var table [numLabels][numBands]int
	var totalPerLabel [numLabels]int

	// Loop over all images in the dataset
	for idx, score := range iForestAnomalyScores {
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
	// RandomForest time	//
	//////////////////////////

	fmt.Println("Starting RandomForest")
	rForest := randomforest.IsolationForest{X: inputData}
	rForest.Train(100000)
	fmt.Println("rForest len", len(rForest.Results))
	fmt.Println("rForest.Results type: ", reflect.TypeOf(rForest.Results))
	fmt.Println("rForest.Results[0] type: ", reflect.TypeOf(rForest.Results[0]))
	fmt.Println("rForest.Results[0]", rForest.Results[0])

	results := rForest.Results
	// Calculate anomaly scores and find min and max scores.
	scores := make([]float64, len(results))
	minScore := float64(results[0][1]) / float64(results[0][0])
	maxScore := minScore
	for i, res := range results {
		scores[i] = float64(res[1]) / float64(res[0])
		if scores[i] < minScore {
			minScore = scores[i]
		}
		if scores[i] > maxScore {
			maxScore = scores[i]
		}
	}

	// Normalize scores to a 0-1 range.
	normalizedScores := normalizeScores(scores, minScore, maxScore)
	fmt.Println("normalizedScores len", len(normalizedScores))
	fmt.Println("Ten normalizedScores", normalizedScores[0:10])

	//////////////////////////
	// CSV time				//
	//////////////////////////

	// Convert slices to maps
	iForestAnomalyScoresMap := make(map[int]float64, len(iForestAnomalyScores))
	normalizedScoresMap := make(map[int]float64, len(normalizedScores))
	for i, v := range iForestAnomalyScores {
		iForestAnomalyScoresMap[i] = v
	}
	for i, v := range normalizedScores {
		normalizedScoresMap[i] = v
	}
	// Convert GoMNIST.Label to int
	labelsInt := make([]int, len(train.Labels))
	for i, label := range train.Labels {
		labelsInt[i] = int(label)
	}

	WriteCSV(iForestAnomalyScoresMap, normalizedScoresMap, labelsInt)
}
