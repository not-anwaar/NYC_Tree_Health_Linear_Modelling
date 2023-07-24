### Linear Modelling of Tree Health in New York City vs. Economic Data ########
###                        Author: Anwaar Hadi                         ########
###
### This R script generates a linear model that can be used to model the 
### health of trees in the New York City area as a function of several factors
### namely the number of other trees in the same area, the location of the tree 
### in question, and the level of economic prosperity of the surrounding area,
### where trees are considered to be in the same area if they are located in the same
### zipcode. For measuring the location of the tree in question, the latitude and
### longitude of the tree are used. As a measure of economic prosperity of an 
### area(zipcode), mean property sale price for the zipcode as well as the 
### number of properties sold in that zipcode are used.
###
### Once coefficients for the model are generated, the model is updated to
### exclude statistically insignificant predictors(as determined by the p-value
### for the coefficient corresponding to the given predictor in the model). The
### root-mean squared error from the projected data to the actual values in the
### dataset are then computed for each zipcode. This data is then exported to 
### CSV format along with several aggregates for the dataset as a whole
###
### The data used in this model is all open-source and is sourced from the NYC
### OpenData portal. 
### 
### The dataset containing street tree data can be found at:
### https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/uvpi-gqnh
###
### The dataset containing the economic data(property sales information) can be
### found at:
### https://data.cityofnewyork.us/City-Government/NYC-Citywide-Annualized-Calendar-Sales-Update/w2pb-icbu
###
### It is worth noting that the street tree data is slightly out-of-date when 
### compared to the property sales data. This is because the Street Tree Dataset
### is historical data obtained from the TreesCount! 2015 Street Tree Census,
### whereas the property sales dataset is updated annually by the NYC Department
### of Finance. The effects of this discrepancy are unknown at this time, however
### are worth investigating in the future.

library(dplyr)

################################################################################
###################### Read Data into R from CSV files #########################
################################################################################

# Read in Street Tree and economic data from csv files.
print("Reading data from csv...")
df_trees <- read.csv("2015_Street_Tree_Census_-_Tree_Data.csv")
df_sales_data <- read.csv("NYC_Citywide_Annualized_Calendar_Sales_Update.csv")
print("Data read. Performing data cleaning and preprocessing...")

################################################################################
################## Preprocess data for use in regression #######################
################################################################################

# Map "Poor", "Fair", and "Good" strings in the health column to integers to
# allow for prediction through regression
df_trees$health[df_trees$health == "Poor"] <- 1.0
df_trees$health[df_trees$health == "Fair"] <- 2.0
df_trees$health[df_trees$health == "Good"] <- 3.0
df_trees$health[df_trees$health == ""] <- 0.0
df_trees$health <- as.numeric(df_trees$health)

# Filter economic and tree data to ensure that both datasets contain the same 
# postcodes
df_sales_data <- df_sales_data %>% 
                 rename(postcode = ZIP.CODE) %>%
                 rename(sale_price = SALE.PRICE) %>%
                 filter(postcode %in% df_trees$postcode)
df_trees <- df_trees %>% filter(postcode %in% df_sales_data$postcode)

# Aggregate sales data in order to obtain the number of properties sold in each
# zipcode as well as the mean sale price of properties sold in each zipcode. 
# These will be used as predictors in the regression.
sales_data_aggregated <- df_sales_data %>%
                          group_by(postcode) %>%
                          summarize(
                            num_properties_sold = n(),
                            mean_sale_price = mean(sale_price, na.rm = TRUE)
                          ) %>%
                          arrange(postcode)

# Merge aggregated sales data into tree data(left-join). Compute the number of 
# trees in each zipcode and generate a new dataset with the data needed for 
# regression(tree_id is included for readability, and will not be used in the 
# regression due to it having little predictive power with respect to tree health)
regression_data <- merge(df_trees, sales_data_aggregated, by = "postcode", all.x = TRUE) %>%
                   group_by(postcode) %>%
                   mutate(num_trees = n()) %>%
                   ungroup() %>%
                   arrange(postcode) %>%
                   select(tree_id, health, postcode, num_trees,
                          num_properties_sold, mean_sale_price,
                          latitude, longitude)

print("Data preprocessing complete. Estimating model coefficients...")

################################################################################
######## Perform a train/test split on the data and estimate coefficients ######
################################################################################

# Perform a 60-40 train/test split of the data. Here, 60% of the data is used 
# for estimating coefficients and the remaining 40% is used to test projections.
# This is done so as to avoid overfitting of the model with respect to the dataset.
rows <- nrow(regression_data)
split <- 0.6
upper_bound <- floor(split * rows)
permuted_regression_data <- regression_data[sample(rows),]
train <- permuted_regression_data[1:upper_bound, ]
test <- permuted_regression_data[(upper_bound+1):rows, ]

# Estimate coefficients of the model using lm() function
tree_model.lm <- lm(health ~ num_trees + num_properties_sold + mean_sale_price +
                    latitude + longitude, data=train)

print("Estimation complete. Here are the coefficients for the model: ")
# print coefficients and summary of model statistics
print(summary(tree_model.lm))

# Note from the above summary that the p-value for the longitude predictor is 
# very large(>0.1), thus implying that this predictor is not statistically
# significant. Thus, the model should be updated to exclude this predictor

print("Longitude found to be not statistically significant(p-value > 0.1")
print("Re-estimating model with exclusion of predictor")
tree_model.lm <- update(tree_model.lm, .~. - longitude, data=train)
print(summary(tree_model.lm))

# All predictors are now statistically significant(p-value <= 0.05).
# The resulting equation would be of the form:
# (health) = w1*(num_trees)+w2*(num_properties_sold)+w3*(mean_sale_price)+
#            w4*(latitude) + w0
# with w1...w4 being the coefficients and w0 being the intercept

################################################################################
################## Calculate Root-Mean Squared Error by Zipcode ################
################################################################################

# Define a function for projecting and calculating errors. This will be used in
# the summarize() call later on to output the full aggregated data and
# projection error for each zipcode
project_and_calculate_error <- function(health, num_trees,
                                        num_properties_sold, mean_sale_price,
                                        latitude) {
  df <- data.frame(health, num_trees, num_properties_sold, mean_sale_price, 
                   latitude)
  projections <- predict(tree_model.lm, newdata = df)
  rmse <- sqrt( sum( (projections - df$health)^2 ) / length(df$health) )
  return(rmse)
}

# Generate an aggregated summary of the data with RMSEs for each zipcode as 
# determined by the model
aggregates_with_errors <- regression_data %>%
                         group_by(postcode) %>%
                         summarize(
                           sample_size = n(),
                           med_health = median(health, na.rm=TRUE),
                           mean_health = mean(health, na.rm=TRUE),
                           standard_deviation = sd(health, na.rm=TRUE),
                           standard_error = standard_deviation / sqrt(sample_size),
                           rmse = project_and_calculate_error(health, num_trees,
                                                              num_properties_sold, mean_sale_price,
                                                              latitude),
                           normalized_rmse = rmse / mean_health
                         )
# Output this data in CSV format
write.csv(aggregates_with_errors, "tree_model_aggregates.csv")