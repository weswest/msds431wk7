package main

import (
	"fmt"

	"github.com/e-XpertSolutions/go-iforest/iforest"
	"github.com/weswest/msds431wk7/mnist"
)

func printData(dataSet *mnist.DataSet, index int) {
	data := dataSet.Data[index]
	fmt.Println(data.Digit)      // print Digit (label)
	mnist.PrintImage(data.Image) // print Image
}

func flattenImages(digitImages []mnist.DigitImage) [][]float64 {
	flattened := make([][]float64, len(digitImages))

	for i, di := range digitImages {
		temp := make([]float64, 0)
		for _, row := range di.Image {
			for _, pixel := range row {
				temp = append(temp, float64(pixel))
			}
		}
		flattened[i] = temp
	}

	return flattened
}

func main() {
	dataSet, err := mnist.ReadTrainSet("./data")
	// or dataSet, err := mnist.ReadTestSet("./mnist")
	if err != nil {
		fmt.Println(err)
		return
	}
	//	fmt.Println(dataSet.N) // number of data
	//	fmt.Println(dataSet.W) // image width [pixel]
	//	fmt.Println(dataSet.H) // image height [pixel]

	flatData := flattenImages(dataSet.Data)
	fmt.Println("Rows: ", len(flatData))
	fmt.Println("Columns: ", len(flatData[0]))

	for i := 0; i < 3; i++ {
		printData(dataSet, i)
		fmt.Println(flatData[i])
	}
	//////////////////////////
	// iForest time			//
	//////////////////////////

	inputData := flatData

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
	threshold := forest.AnomalyBound
	anomalyScores := forest.AnomalyScores
	labelsTest := forest.Labels

	//to get information about new instances pass them to the Predict function
	// to speed up computation use concurrent version of Predict
	// var newData [][]float64
	// newData = loadData("someNewInstances")
	// labels, scores := forest.Predict(newData)
}
