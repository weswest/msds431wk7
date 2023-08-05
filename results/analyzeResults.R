# compare the anomaly scores from Python and R
digitLabels = read.csv("results/labels.csv")
scoresSolitudeR = read.csv("results/solitudeRScores.csv")
scoresIsotreeR = read.csv("results/isotreeRScores.csv")
scoresPython = read.csv("results/pythonScores.csv")
scoresGo = read.csv("results/goIForestScores.csv")
  
# merge the scoring data  
analyzeData <- data.frame("digitLabel" = digitLabels$digitLabel,
	"scoreSolitudeR" = scoresSolitudeR$iforestRScore.anomaly_score,
	"scoreIsotreeR" = scoresIsotreeR$isotreeRScore,
	"scorePython" = scoresPython$iforestPythonScore,
	"scoreGo" = scoresGo$anomalyScore)

# Note that distributions of anomaly scores have different shapes
# Are there hyperparameter settings that may bring the 
# Python and R results closer together?
pdf(file = "results/fig-python-anomaly-scores.pdf", width = 11, height = 8.5)
with(analyzeData, plot(density(scorePython)))
dev.off()

pdf(file = "results/fig-r-solitude-anomaly-scores.pdf", width = 11, height = 8.5)
with(analyzeData, plot(density(scoreSolitudeR)))
dev.off()

pdf(file = "results/fig-r-isotree-anomaly-scores.pdf", width = 11, height = 8.5)
with(analyzeData, plot(density(scoreIsotreeR)))
dev.off()

pdf(file = "results/fig-scatterplot-solitude-anomaly-scores.pdf", width = 11, height = 8.5)
with(analyzeData, plot(scorePython,scoreSolitudeR))
title(paste("Correlation between Python and R solitude anomaly scores:",
	as.character(round(with(analyzeData,cor(scorePython,scoreSolitudeR)),digits = 2))))
dev.off()

pdf(file = "results/fig-scatterplot-isotree-anomaly-scores.pdf", width = 11, height = 8.5)
with(analyzeData, plot(scorePython,scoreIsotreeR))
title(paste("Correlation between Python and R isotree anomaly scores:",
	as.character(round(with(analyzeData,cor(scorePython,scoreIsotreeR)),digits = 2))))
dev.off()


# Go results

pdf(file = "results/fig-go-itree-anomaly-scores.pdf", width = 11, height = 8.5)
with(analyzeData, plot(density(scoreGo)))
dev.off()

pdf(file = "results/fig-scatterplot-go-x-python-anomaly-scores.pdf", width = 11, height = 8.5)
with(analyzeData, plot(scoreGo,scorePython))
title(paste("Correlation between Go I-Tree and Python anomaly scores:",
	as.character(round(with(analyzeData,cor(scoreGo, scorePython)),digits = 2))))
dev.off()

pdf(file = "results/fig-scatterplot-go-x-risotree-anomaly-scores.pdf", width = 11, height = 8.5)
with(analyzeData, plot(scoreGo,scoreIsotreeR))
title(paste("Correlation between Go I-Tree and rIsotree anomaly scores:",
	as.character(round(with(analyzeData,cor(scoreGo, scoreIsotreeR)),digits = 2))))
dev.off()

pdf(file = "results/fig-scatterplot-go-x-solitude-anomaly-scores.pdf", width = 11, height = 8.5)
with(analyzeData, plot(scoreGo,scoreSolitudeR))
title(paste("Correlation between Go I-Tree and r Solitude anomaly scores:",
	as.character(round(with(analyzeData,cor(scoreGo, scoreSolitudeR)),digits = 2))))
dev.off()
