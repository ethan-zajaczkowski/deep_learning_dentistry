import torch
import torch.nn as nn

# Define embedding dimensions and category sizes
embedding_dim = 8  # Dimension for each embedding vector
num_countries = 3  # Suppose we have 3 countries
num_cities = 5     # Suppose we have 5 cities
num_genders = 2    # Suppose we have 2 gender categories

# Create embedding layers for each categorical variable
country_embed = nn.Embedding(num_countries, embedding_dim)
city_embed = nn.Embedding(num_cities, embedding_dim)
gender_embed = nn.Embedding(num_genders, embedding_dim)

# For the continuous variable (age), use a linear layer to project from 1 to embedding_dim
age_embed = nn.Linear(1, embedding_dim)

# Example data for a batch of 2 samples:
# Each sample consists of: [country, city, gender, age]
countries = torch.tensor([0, 1])           # Example indices for countries
cities = torch.tensor([1, 3])              # Example indices for cities
genders = torch.tensor([0, 1])             # Example indices for gender
ages = torch.tensor([[30.0], [25.0]])       # Age values as a 2D tensor (batch_size x 1)

# Get embeddings for each variable
country_vec = country_embed(countries)  # Shape: [2, embedding_dim]
city_vec = city_embed(cities)           # Shape: [2, embedding_dim]
gender_vec = gender_embed(genders)      # Shape: [2, embedding_dim]
age_vec = age_embed(ages)               # Shape: [2, embedding_dim]

# Concatenate the embeddings along the last dimension
combined_tensor = torch.cat([country_vec, city_vec, gender_vec, age_vec], dim=1)

print("Combined tensor shape:", combined_tensor.shape)
print(combined_tensor)