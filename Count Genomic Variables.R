# Install necessary package
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if (!requireNamespace("GEOquery", quietly = TRUE))
  BiocManager::install("GEOquery")

# Load GEOquery library
library(GEOquery)

# List of GEO dataset IDs
geo_ids <- c("GSE29013", "GSE6044", "GSE14814", "GSE7880", 
             "GSE39279", "GSE50081", "GSE39345", "GSE42127", 
             "GSE37745", "GSE47115")

# Initialize a data frame to store results
results <- data.frame(
  GEO_ID = character(),
  Num_Genes = integer(),
  Num_Samples = integer(),
  Platform = character(),
  stringsAsFactors = FALSE
)

# Loop through GEO IDs
for (geo_id in geo_ids) {
  cat("Processing:", geo_id, "\n")
  
  # Download GEO dataset
  gse <- tryCatch(
    getGEO(geo_id, GSEMatrix = TRUE),
    error = function(e) {
      message("Error loading dataset: ", geo_id)
      return(NULL)
    }
  )
  
  # Skip if dataset couldn't be loaded
  if (is.null(gse)) next
  
  # Extract first dataset (if multiple are available)
  gse <- gse[[1]]
  
  # Get the number of genes and samples
  num_genes <- nrow(exprs(gse))
  num_samples <- ncol(exprs(gse))
  
  # Get platform information
  platform <- annotation(gse)
  
  # Append to results
  results <- rbind(results, data.frame(
    GEO_ID = geo_id,
    Num_Genes = num_genes,
    Num_Samples = num_samples,
    Platform = platform,
    stringsAsFactors = FALSE
  ))
}

# Display the results
print(results)

# Save results to a CSV file
write.csv(results, "geo_dataset_summary.csv", row.names = FALSE)

cat("Analysis complete. Results saved to 'geo_dataset_summary.csv'.\n")