package main

import (
	"testing"

	"github.com/petar/GoMNIST"
)

func TestDataReadin(t *testing.T) {
	train, _, err := GoMNIST.Load("./data")
	if err != nil {
		t.Errorf("No data found")
	}

	if len(train.Images) != 60000 {
		t.Errorf("Expected 60000 images, got %d", len(train.Images))
	}

	if len(train.Images[0]) != 784 {
		t.Errorf("Expected 784 pixels per image, got %d", len(train.Images[0]))
	}
}

func TestNormalizeScores(t *testing.T) {
	cases := []struct {
		name   string
		input  []float64
		min    float64
		max    float64
		output []float64
	}{
		{
			"case 1",
			[]float64{1, 2, 3, 4, 5},
			1,
			5,
			[]float64{0, 0.25, 0.5, 0.75, 1},
		},
		{
			"case 2",
			[]float64{10, 20, 30, 40, 50},
			10,
			50,
			[]float64{0, 0.25, 0.5, 0.75, 1},
		},
		{
			"case 3",
			[]float64{0, 0.25, 0.5, 0.75, 1},
			0,
			1,
			[]float64{0, 0.25, 0.5, 0.75, 1},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			output := normalizeScores(tc.input, tc.min, tc.max)
			if !equalFloat64Slice(output, tc.output) {
				t.Errorf("Expected %v, got %v", tc.output, output)
			}
		})
	}
}

// Helper function to check if two slices of float64 are equal
func equalFloat64Slice(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
