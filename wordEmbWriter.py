import torch
import os

wordEmbWriterPath = "/cluster/datastore/aniketag/newWordStylist/WordStylist/wordEmbWriter2.pt"

if os.path.isfile(wordEmbWriterPath):
    print("\n\t Embedding dict loaded!!!")
    wordEmbWriter = torch.load(wordEmbWriterPath)
else:
    print("\n\t Dictionary empty!!!!")
    wordEmbWriter = {}

# Function to calculate correlation among tensors
def compute_correlation(tensors):
    # Stack all tensors
    stacked_tensors = torch.stack(tensors)  # shape: (num_tensors, 1, 10, 320)
    
    # Flatten each tensor along dimensions 1, 2, and 3 (keeping the first dimension as batch)
    flattened_tensors = stacked_tensors.view(stacked_tensors.size(0), -1)  # shape: (num_tensors, 10*320)
    
    # Compute the covariance matrix
    cov_matrix = torch.cov(flattened_tensors.T)
    
    # Compute standard deviations
    std_devs = torch.sqrt(torch.diag(cov_matrix))
    
    # Compute the correlation matrix by normalizing the covariance matrix
    correlation_matrix = cov_matrix / torch.outer(std_devs, std_devs)
    
    # Loop through each pair and print the correlation one by one
    num_tensors = flattened_tensors.size(0)
    for i in range(num_tensors):
        for j in range(i + 1, num_tensors):  # Only consider pairs once (i, j) where i < j
            print(f"\n\t\tCorrelation between tensor {i} and tensor {j}: {correlation_matrix[i, j].item()}")

    
    
    
    return correlation_matrix

print("\n\t ALPHA:",wordEmbWriter.keys())

for writerID in wordEmbWriter.keys():
    if len(wordEmbWriter[writerID]) >= 1:
        valWriter = wordEmbWriter[writerID]
        
        print("\n\t writerID:",writerID)
        val = []
        for key2 in valWriter.keys():
            print("\n\t\t key2:", key2, "\t valWriter[key2][0].shape:", valWriter[key2][0].shape, len(valWriter[key2]))
            
            # Collect all tensors for this specific key2
            tensors_for_key2 = valWriter[key2]  # This is a list of tensors for the current key2

            # Compute the correlation among these tensors
            correlation_matrix = compute_correlation(tensors_for_key2)
            
            # Print or store the correlation matrix
            #print(f"\n\t\t\t Correlation Matrix for key2 = {key2}")
            #print("\n\t\t\t",correlation_matrix)
