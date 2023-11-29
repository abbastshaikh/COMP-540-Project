library(spatialRF)
library(rspatial)

random.seed <- 1

setwd("C:\Users\mathw\Desktop\College\Fall 23\COMP 540\Project\COMP-540-Project")
data <- read.csv("data/for_model.csv")
data <- head(data, 1000)

predictor.variable.names <- colnames(data)[4:42]
xy <- data[, c("Longitude", "Latitude")]
distance_matrix <- as.matrix(dist(xy))

for (dependent.variable.name in colnames(data)[43:45]) { 
  spatial.model <- spatialRF::rf_spatial(
    data = data,
    dependent.variable.name = dependent.variable.name,
    predictor.variable.names = predictor.variable.names,
    distance.matrix = distance_matrix,
    seed = random.seed
  )
}
