# Load necessary libraries
library(data.table)
library(ggplot2)
library(reshape2)

# Sigmoid activation function
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

# Derivative of the sigmoid function
sigmoid_derivative <- function(x) {
  x * (1 - x)
}

# Scaling function
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}


# Load the dataset
my_data <- fread("D:\\MyWorkSpace\\dataset\\cross.csv")

# Scale the data
scaled <- normalize(my_data)
scaled_data <- scaled
maxs <- scaled$maxs
mins <- scaled$mins

# Extract input and output
x <- as.matrix(scaled_data[, 1:2])
y <- as.matrix(scaled_data[, 3:4])

# Initialize parameters
set.seed(42)
input_layer_neurons <- ncol(x)
hidden_layer_neurons <- 32
output_layer_neurons <- ncol(y)

# Training parameters
learning_rate <- 0.1
momentum_rate <- 0.9
epochs <- 10000
k_folds <- 10

# K-fold cross-validation
folds <- sample(rep(1:k_folds, length.out = nrow(x)))

loss_history <- vector("list", k_folds)
fold_results <- numeric(k_folds)
predicted_outputs <- matrix(0, nrow = nrow(x), ncol = output_layer_neurons)

for (k in 1:k_folds) {
  cat("Fold", k, "\n")
  
  # Split data into training and validation sets
  test_indices <- which(folds == k, arr.ind = TRUE)
  X_train <- x[-test_indices, ]
  y_train <- y[-test_indices, ]
  X_test <- x[test_indices, ]
  y_test <- y[test_indices, ]
  
  # Initialize weights, biases, and momentum terms
  weights_input_hidden <- matrix(runif(input_layer_neurons * hidden_layer_neurons, min = -1, max = 1), nrow = input_layer_neurons)
  weights_hidden_output <- matrix(runif(hidden_layer_neurons * output_layer_neurons, min = -1, max = 1), nrow = hidden_layer_neurons)
  
  bias_hidden <- runif(hidden_layer_neurons, min = -1, max = 1)
  bias_output <- runif(output_layer_neurons, min = -1, max = 1)
  
  momentum_weights_input_hidden <- matrix(0, nrow = input_layer_neurons, ncol = hidden_layer_neurons)
  momentum_weights_hidden_output <- matrix(0, nrow = hidden_layer_neurons, ncol = output_layer_neurons)
  momentum_bias_hidden <- rep(0, hidden_layer_neurons)
  momentum_bias_output <- rep(0, output_layer_neurons)
  
  # Training the neural network
  fold_loss_history <- numeric(epochs)
  
  for (epoch in 1:epochs) {
    # Forward propagation
    hidden_layer_activation <- X_train %*% weights_input_hidden + matrix(rep(bias_hidden, nrow(X_train)), nrow = nrow(X_train), byrow = TRUE)
    hidden_layer_output <- sigmoid(hidden_layer_activation)
    
    output_layer_activation <- hidden_layer_output %*% weights_hidden_output + matrix(rep(bias_output, nrow(hidden_layer_output)), nrow = nrow(hidden_layer_output), byrow = TRUE)
    predicted_output <- sigmoid(output_layer_activation)
    
    # Compute loss (Mean Squared Error)
    loss <- mean((y_train - predicted_output) ^ 2)
    fold_loss_history[epoch] <- loss
    
    # Print error at every epoch
    if (epoch %% 1000 == 0) {
      cat("Epoch", epoch, "Loss:", loss, "\n")
    }
    
    # Backward propagation
    error <- y_train - predicted_output
    d_predicted_output <- error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer <- d_predicted_output %*% t(weights_hidden_output)
    d_hidden_layer <- error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Compute gradients
    gradient_weights_hidden_output <- t(hidden_layer_output) %*% d_predicted_output
    gradient_bias_output <- colSums(d_predicted_output)
    
    gradient_weights_input_hidden <- t(X_train) %*% d_hidden_layer
    gradient_bias_hidden <- colSums(d_hidden_layer)
    
    # Update weights and biases using gradients and momentum
    momentum_weights_hidden_output <- momentum_rate * momentum_weights_hidden_output + learning_rate * gradient_weights_hidden_output
    weights_hidden_output <- weights_hidden_output + momentum_weights_hidden_output
    
    momentum_bias_output <- momentum_rate * momentum_bias_output + learning_rate * gradient_bias_output
    bias_output <- bias_output + momentum_bias_output
    
    momentum_weights_input_hidden <- momentum_rate * momentum_weights_input_hidden + learning_rate * gradient_weights_input_hidden
    weights_input_hidden <- weights_input_hidden + momentum_weights_input_hidden
    
    momentum_bias_hidden <- momentum_rate * momentum_bias_hidden + learning_rate * gradient_bias_hidden
    bias_hidden <- bias_hidden + momentum_bias_hidden
  }
  
  # Evaluate the model on the validation set
  hidden_layer_activation <- X_test %*% weights_input_hidden + matrix(rep(bias_hidden, nrow(X_test)), nrow = nrow(X_test), byrow = TRUE)
  hidden_layer_output <- sigmoid(hidden_layer_activation)
  
  output_layer_activation <- hidden_layer_output %*% weights_hidden_output + matrix(rep(bias_output, nrow(hidden_layer_output)), nrow = nrow(hidden_layer_output), byrow = TRUE)
  predicted_output <- sigmoid(output_layer_activation)
  
  predicted_outputs[test_indices, ] <- predicted_output
  
  test_loss <- mean((y_test - predicted_output) ^ 2)
  fold_results[k] <- test_loss
  loss_history[[k]] <- fold_loss_history
  cat("Fold", k, "Test Loss:", test_loss, "\n")
}

# Average test loss over all folds
mean_test_loss <- mean(fold_results)
cat("Mean Test Loss over all folds:", mean_test_loss, "\n")

# Plot the loss history for the first fold (as an example)
plot(loss_history[[1]], type = "l", col = "blue", lwd = 2, xlab = "Epochs", ylab = "Loss (MSE)", main = "Training Loss Over Epochs")
lines(loss_history[[2]], col = "red")
lines(loss_history[[3]], col = "purple")
lines(loss_history[[4]], col = "orange")
lines(loss_history[[5]], col = "yellow")
lines(loss_history[[6]], col = "pink")
lines(loss_history[[7]], col = "green")
lines(loss_history[[8]], col = "aquamarine")
lines(loss_history[[9]], col = "lightblue1")
lines(loss_history[[10]], col = "slateblue")

legend(x = "topright", legend=c("Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5","Fold 6", "Fold 7", "Fold 8", "Fold 9", "Fold 10"),
       fill = c("blue", "red", "purple", "orange", "yellow", "pink", "green", "aquamarine", "lightblue1", "slateblue")
)
# Plot actual vs. predicted values for the first fold
# test_length <- seq(1, length(y), 1)
# plot(test_length, y, col = "red", type = "p", xlab = "Index", ylab = "Value", main = "Actual vs. Predicted Values")
# points(test_length, predicted_outputs, col = "blue")
# legend("topright", legend = c("Actual", "Predicted"), col = c("red", "blue"), pch = 1)

predicted <- predicted_outputs * (max(my_data) - min(my_data)) + min(my_data)

for (i in 1:nrow(predicted)){
  if(predicted[i,1]>predicted[i,2]){
    predicted[i,1] <- 1
    predicted[i,2] <- 0
  }
  else{
    predicted[i,1] <- 0
    predicted[i,2] <- 1
  }
  i+1
}
   
# Print the results
results <- data.frame(
  Actual = my_data[,3:4],
  Predicted = predicted
)

label_levels <- c("(1,0)", "(0,1)")
conf_matrix <- matrix(0, nrow = 2, ncol = 2, dimnames = list("Predicted" = label_levels, "Actual" = label_levels))


for(i in 1:nrow(my_data)){
  if(results[i,1] == results[i,3]){
    if(results[i,1]==1){
      conf_matrix[1,1] <- conf_matrix[1,1] + 1
    }
    else if(results[i,1]==0){
      conf_matrix[2,2] <- conf_matrix[2,2]+1
    }
  }
  else{
    if(results[i,2]==1){
      conf_matrix[1,2] <- conf_matrix[1,2] + 1
    }
    else if(results[i,2]==0){
      conf_matrix[2,1] <- conf_matrix[2,1]+1
    }
  }
  i+1
}

melted_data <- melt(conf_matrix)
head(melted_data)

heatmap_plot <- ggplot(melted_data, aes(x = Actual, y = Predicted, fill= value)) + 
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") + # Set color gradient
  theme_minimal() + # Set theme
  labs(x = "Actual", y = "Predicted", title = "Confusion Matrix")

heatmap_plot_with_values <- heatmap_plot +
  geom_text(aes(label = value), color = "black", size = 12)

print(heatmap_plot_with_values)

TP <- conf_matrix[1,1]
FP <- conf_matrix[1,2]
FN <- conf_matrix[2,1]
TN <- conf_matrix[2,2]

acc <- (TP+TN) / (TP + FP + FN + TN)
precision <- TP / (TP + FP)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)

print(conf_matrix)
cat(" Accuracy : ", acc, "\n",
"Sensitivity : ", sensitivity, "\n",
"Specificity : ", specificity, "\n",
"Precision : ", precision)



